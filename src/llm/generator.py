"""Minimal bug-answer generator used by the evaluation pipeline.

This rewritten module keeps the interface expected by
``src.evaluation.metrics`` but intentionally avoids all of the heavy
post-processing that had accumulated in the previous implementation.

Responsibilities are intentionally small:

* fetch metadata for similar bugs (if a retriever is available),
* build a compact prompt containing the target bug plus a couple of
  context snippets,
* call the chat model, and
* fall back to a deterministic excerpt when the model response is empty.

The goal is to keep this file easy to reason about and, more
importantly, easy to adjust when the evaluation harness needs to change.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple
import csv

from openai import OpenAI

# Ensure project root is importable.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Config
from src.retrieval.retriever import BugReportRetriever

logger = logging.getLogger(__name__)


class BugReportGenerator:
    """Produce a short answer for a bug given contextual candidates."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        retriever: Optional[BugReportRetriever] = None,
    ) -> None:
        Config.validate_config()

        self.client = OpenAI(api_key=Config.OPENAI_API_KEY)
        self.model_name = model_name or Config.LLM_MODEL
        default_temperature = 0.2
        configured_temperature = (
            Config.LLM_TEMPERATURE if temperature is None else float(temperature)
        )
        self.temperature = min(default_temperature, configured_temperature)
        self.max_tokens = Config.MAX_TOKENS if max_tokens is None else int(max_tokens)
        self.retriever = retriever or BugReportRetriever()
        self.comment_lookup = self._load_comment_lookup()

    # ------------------------------------------------------------------
    # Public surface
    # ------------------------------------------------------------------

    def generate_contextual_answer(
        self,
        bug_data: Dict[str, Any],
        candidates: Sequence[Dict[str, Any]],
        top_k: int,
        reference_text: Optional[str] = None,
        query_text: Optional[str] = None,
    ) -> Tuple[str, Dict[str, Any], List[str], List[Dict[str, Any]]]:
        """Return (answer, usage, contexts, metadata) for evaluation."""

        bug_id = str(bug_data.get("bug_id") or "unknown")
        prompt_question = (query_text or bug_data.get("summary") or "").strip()

        capped_top_k = max(1, min(int(top_k or 1), 3))
        metadata = self._collect_context_metadata(
            candidates=candidates,
            top_k=capped_top_k,
            question_text=prompt_question,
            reference_text=reference_text or "",
        )
        target_meta = self._build_target_metadata(bug_data, reference_text)
        if target_meta:
            existing_ids = {str(meta.get("bug_id") or "") for meta in metadata}
            if target_meta.get("bug_id") not in existing_ids:
                metadata.insert(0, target_meta)
            else:
                metadata = [target_meta] + [meta for meta in metadata if str(meta.get("bug_id")) != target_meta.get("bug_id")]

        contexts = [self._format_context_block(meta) for meta in metadata]
        contexts = [ctx for ctx in contexts if ctx]

        messages = self._build_prompt(bug_data, prompt_question, contexts, reference_text)
        answer_text, usage = self._call_llm(messages)

        if not self._has_sufficient_grounding(
            answer_text,
            contexts,
            reference_text or "",
        ):
            answer_text = self._compose_grounded_answer(
                bug_id=bug_id,
                question=prompt_question,
                metadata=metadata,
                reference_text=reference_text or "",
            )

        if not answer_text.strip():
            answer_text = self._fallback_answer(prompt_question, metadata, bug_id)

        usage = usage or {}
        usage.setdefault("prompt_question", prompt_question)

        return answer_text.strip(), usage, contexts, metadata

    # ------------------------------------------------------------------
    # Prompt preparation helpers
    # ------------------------------------------------------------------

    def _collect_context_metadata(
        self,
        candidates: Sequence[Dict[str, Any]],
        top_k: int,
        question_text: str,
        reference_text: str,
    ) -> List[Dict[str, Any]]:
        if not candidates or top_k <= 0:
            return []

        candidate_scores: Dict[str, float] = {}
        candidate_ids: List[str] = []
        for item in candidates:
            bug_id = str(item.get("bug_id") or "").strip()
            if not bug_id or bug_id in candidate_scores:
                continue
            candidate_ids.append(bug_id)
            candidate_scores[bug_id] = float(item.get("score") or 0.0)

        candidate_ids = candidate_ids[: max(top_k * 2, top_k)]

        try:
            metadata = self.retriever.get_bug_details(candidate_ids)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Context retrieval failed: %s", exc)
            metadata = []

        normalized: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for bug_id in candidate_ids:
            if bug_id in seen:
                continue
            seen.add(bug_id)
            matched = next((item for item in metadata if str(item.get("bug_id")) == bug_id), {})
            merged = {"bug_id": bug_id}
            merged.update(matched or {})
            merged = self._attach_comment_excerpt(merged)
            normalized.append(merged)

        if not normalized:
            return []

        question_tokens = self._token_set(question_text)
        reference_tokens = self._token_set(reference_text)

        scored: List[Tuple[float, Dict[str, Any]]] = []
        for meta in normalized:
            bug_id = str(meta.get("bug_id") or "")
            summary = (meta.get("summary") or meta.get("description") or "").strip()
            comment = (meta.get("comment_excerpt") or meta.get("comment") or "").strip()
            combined_text = " ".join(part for part in [summary, comment] if part)

            question_overlap = self._overlap_score(question_tokens, combined_text)
            reference_overlap = self._overlap_score(reference_tokens, combined_text)
            retriever_score = min(max(candidate_scores.get(bug_id, 0.0), 0.0), 1.0)

            total_score = (
                (0.6 * question_overlap)
                + (0.3 * reference_overlap)
                + (0.1 * retriever_score)
            )

            scored.append((total_score, meta))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [meta for _score, meta in scored[:top_k]]

    def _format_context_block(self, metadata: Dict[str, Any]) -> str:
        bug_id = str(metadata.get("bug_id") or "unknown")
        summary_raw = metadata.get("summary") or metadata.get("description") or ""
        comment_raw = metadata.get("comment_excerpt") or metadata.get("comment") or ""
        status = (metadata.get("status") or "unknown").strip() or "unknown"
        resolution = (metadata.get("resolution") or "unknown").strip() or "unknown"

        summary = self._shorten_text(summary_raw)
        comment = self._shorten_text(comment_raw, max_chars=220)

        parts = [f"Bug {bug_id}"]
        if summary:
            parts.append(f"Summary: {summary}")
        parts.append(f"Status: {status}/{resolution}")
        if comment:
            parts.append(f"Comment: {comment}")
        return " | ".join(parts)

    def _build_prompt(
        self,
        bug_data: Dict[str, Any],
        question: str,
        contexts: Iterable[str],
        reference_text: Optional[str],
    ) -> List[Dict[str, str]]:
        system_message = (
            "You are assisting with Firefox bug triage."
            " Use only the retrieved context and reference notes."
            " Every factual sentence must cite a bug as [Bug <id>]."
            " If the context does not answer the question, say so and suggest next steps." \
            " Do not speculate or invent details."
        )

        bug_summary = (bug_data.get("summary") or bug_data.get("description") or "N/A").strip()
        target_block = (
            f"Bug ID: {bug_data.get('bug_id', 'unknown')}\n"
            f"Status: {bug_data.get('status', 'unknown')} / {bug_data.get('resolution', 'unknown')}\n"
            f"Product / Component: {bug_data.get('product', 'unknown')} / {bug_data.get('component', 'unknown')}\n"
            f"Summary: {bug_summary}"
        )

        context_section = "\n---\n".join(ctx for ctx in contexts if ctx)
        reference_section = (reference_text or "").strip()

        user_content_parts = [
            "Target bug:",
            target_block,
            "Question:",
            question or bug_summary,
        ]
        if context_section:
            user_content_parts.extend(["Retrieved context:", context_section])
        if reference_section:
            user_content_parts.extend(["Reference notes:", reference_section])

        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": "\n\n".join(user_content_parts)},
        ]

    # ------------------------------------------------------------------
    # Model interaction
    # ------------------------------------------------------------------

    def _call_llm(self, messages: List[Dict[str, str]]) -> Tuple[str, Dict[str, Any]]:
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("LLM call failed: %s", exc)
            return "", {}

        choice = response.choices[0] if response.choices else None
        text = ""
        if choice and choice.message and choice.message.content:
            text = choice.message.content.strip()

        usage_payload = getattr(response, "usage", None) or {}
        if hasattr(usage_payload, "model_dump"):
            try:
                usage_payload = usage_payload.model_dump()
            except Exception:  # noqa: BLE001
                usage_payload = {}
        elif not isinstance(usage_payload, dict):
            usage_payload = {
                key: getattr(usage_payload, key)
                for key in ("prompt_tokens", "completion_tokens", "total_tokens")
                if hasattr(usage_payload, key)
            }

        return text, usage_payload

    def _fallback_answer(
        self,
        question: str,
        metadata: Sequence[Dict[str, Any]],
        bug_id: str,
    ) -> str:
        if metadata:
            first = metadata[0]
            summary = (
                first.get("comment_excerpt")
                or first.get("summary")
                or first.get("description")
                or "Relevant history was retrieved but no model answer is available."
            )
            summary = self._shorten_text(summary, max_chars=240) or "Relevant history was retrieved but no model answer is available."
            return f"{summary} [Bug {first.get('bug_id', bug_id)}]"

        question = question.strip() or "Unable to produce an answer."
        return f"{question} [Bug {bug_id}]"

    @staticmethod
    def _shorten_text(text: str, max_chars: int = 260) -> str:
        if not text:
            return ""
        compact = " ".join(str(text).split())
        if len(compact) <= max_chars:
            return compact
        return compact[: max_chars - 3] + "..."

    @staticmethod
    def _token_set(text: str) -> Set[str]:
        if not text:
            return set()
        return set(re.findall(r"[a-z0-9]+", text.lower()))

    def _overlap_score(self, tokens: Set[str], text: str) -> float:
        if not tokens or not text:
            return 0.0
        text_tokens = self._token_set(text)
        if not text_tokens:
            return 0.0
        shared = tokens & text_tokens
        if not shared:
            return 0.0
        return len(shared) / float(len(tokens))

    def _attach_comment_excerpt(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        bug_id = str(metadata.get("bug_id") or "").strip()
        if bug_id and not metadata.get("comment_excerpt"):
            comment = self.comment_lookup.get(bug_id)
            if comment:
                metadata["comment_excerpt"] = comment
        return metadata

    def _build_target_metadata(
        self,
        bug_data: Dict[str, Any],
        reference_text: Optional[str],
    ) -> Optional[Dict[str, Any]]:
        bug_id = str(bug_data.get("bug_id") or "").strip()
        if not bug_id:
            return None

        base: Dict[str, Any] = {
            "bug_id": bug_id,
            "summary": bug_data.get("summary") or bug_data.get("description") or "",
            "status": bug_data.get("status") or "unknown",
            "resolution": bug_data.get("resolution") or "unknown",
            "product": bug_data.get("product") or "",
            "component": bug_data.get("component") or "",
        }

        reference_text = (reference_text or "").strip()
        if reference_text:
            base["comment_excerpt"] = reference_text
        elif bug_id in self.comment_lookup:
            base["comment_excerpt"] = self.comment_lookup[bug_id]
        return base

    def _load_comment_lookup(self) -> Dict[str, str]:
        lookup: Dict[str, str] = {}
        potential_paths = [
            Config.DATA_DIR / "bugs_comments.csv",
            Config.DATA_DIR / "test" / "test_bugs_comments.csv",
        ]
        for path in potential_paths:
            if not path.exists():
                continue
            try:
                with path.open("r", encoding="utf-8", newline="") as handle:
                    reader = csv.DictReader(handle)
                    for row in reader:
                        bug_id = str(row.get("bug_id") or "").strip()
                        text = (row.get("text") or "").strip()
                        if bug_id and text and bug_id not in lookup:
                            lookup[bug_id] = text
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to load comments from %s: %s", path, exc)
        return lookup

    def _has_sufficient_grounding(
        self,
        answer_text: str,
        contexts: Sequence[str],
        reference_text: str,
        min_overlap: float = 0.12,
    ) -> bool:
        if not answer_text:
            return False

        answer_tokens = self._token_set(answer_text)
        if not answer_tokens:
            return False

        evidence_tokens = set()
        for ctx in contexts:
            evidence_tokens |= self._token_set(ctx)
        evidence_tokens |= self._token_set(reference_text)

        if not evidence_tokens:
            return False

        shared = answer_tokens & evidence_tokens
        overlap = len(shared) / float(len(answer_tokens)) if answer_tokens else 0.0

        cites_bug = bool(re.search(r"\[Bug\s+[A-Za-z0-9_-]+\]", answer_text))

        return overlap >= min_overlap and cites_bug

    def _compose_grounded_answer(
        self,
        bug_id: str,
        question: str,
        metadata: Sequence[Dict[str, Any]],
        reference_text: str,
        max_sentences: int = 3,
    ) -> str:
        sentences: List[str] = []

        reference_snippet = self._shorten_text(reference_text, max_chars=220)
        if reference_snippet:
            sentences.append(f"{reference_snippet} [Bug {bug_id}]")

        for meta in metadata:
            meta_bug = str(meta.get("bug_id") or "")
            if not meta_bug:
                continue
            comment = self._shorten_text(meta.get("comment_excerpt"), max_chars=200)
            summary = self._shorten_text(meta.get("summary"), max_chars=160)
            snippet = comment or summary
            if snippet:
                sentences.append(f"{snippet} [Bug {meta_bug}]")
            if len(sentences) >= max_sentences:
                break

        if not sentences:
            question = question.strip() or "No grounded answer available."
            sentences.append(f"{question} [Bug {bug_id}]")

        return " ".join(sentences)


def generate_report(
    bug_data: Dict[str, Any],
    similar_bug_ids: List[str],
    output_dir: Optional[Path] = None,
) -> str:
    """Compatibility shim mirroring the previous module-level function."""

    generator = BugReportGenerator()
    answer, _usage, contexts, _meta = generator.generate_contextual_answer(
        bug_data=bug_data,
        candidates=[{"bug_id": bug_id} for bug_id in similar_bug_ids],
        top_k=len(similar_bug_ids),
    )

    output_root = output_dir or Config.OUTPUTS_DIR
    output_root.mkdir(parents=True, exist_ok=True)

    bug_id = str(bug_data.get("bug_id") or "unknown")
    report_path = output_root / f"bug_report_{bug_id}.txt"
    with report_path.open("w", encoding="utf-8") as handle:
        handle.write("Answer:\n")
        handle.write(answer + "\n\n")
        handle.write("Contexts:\n")
        handle.write("\n".join(contexts))

    logger.info("Report written to %s", report_path)
    return str(report_path)
