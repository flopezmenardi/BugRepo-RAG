"""
RAG quality evaluation script leveraging ragas metrics (fixed & tougher grounding).

Key fixes:
  * Send Chat Completions the correct messages shape (system+user list).
  * Robust retry + extractive fallback when LLM fails.
  * Enriched, re-ranked contexts (default top_k=8) for better grounding.
  * Still uses strict extractive prompt.

Usage:
    python -m src.evaluation.metrics --limit 10 --top-k-contexts 8
"""

import argparse
import csv
import math
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )
except ImportError as exc:
    raise ImportError(
        "No se encontró ragas o datasets. Instalá dependencias con "
        "`pip install ragas datasets`."
    ) from exc

# Project path bootstrap
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Config
from src.embeddings.embedder import BugReportEmbedder
from src.llm.generate_report import generate_report
from src.llm.generator import BugReportGenerator
from src.retrieval.retriever import BugReportRetriever

logger = logging.getLogger(__name__)

# Test dataset paths
TEST_DATA_DIR = Config.DATA_DIR / "test"
TEST_BUGS_CSV = TEST_DATA_DIR / "test_bugs.csv"
TEST_COMMENTS_CSV = TEST_DATA_DIR / "test_bugs_comments.csv"


# ---------------------------
# CSV LOADING & GROUND TRUTH
# ---------------------------

def load_bug_rows(bugs_csv: Path, limit: Optional[int] = None) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with bugs_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if not (row.get("summary") or "").strip():
                continue
            rows.append(row)
            if limit and len(rows) >= limit:
                break
    logger.info("Loaded %d bugs from %s", len(rows), bugs_csv)
    return rows


def load_ground_truth(path: Path) -> Dict[str, str]:
    """Load ground-truth reference texts.

    If `path` is a directory, load files named like
    `bug_<id>_triage_report.txt` and map `<id>` -> file contents.
    If `path` is a file (CSV), fall back to the previous behaviour and
    return an empty mapping (CSV-based comment loading is handled elsewhere).
    """
    mapping: Dict[str, str] = {}
    try:
        if path.is_dir():
            for child in sorted(path.iterdir()):
                if not child.is_file():
                    continue
                name = child.name
                # match files like bug_1938695_triage_report.txt
                import re

                m = re.match(r"bug_([A-Za-z0-9_-]+)_triage_report\.txt$", name)
                if not m:
                    continue
                bug_id = m.group(1)
                try:
                    text = child.read_text(encoding="utf-8").strip()
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Failed to read ground truth %s: %s", child, exc)
                    continue
                if text:
                    mapping[str(bug_id)] = text
            logger.info("Loaded %d ground-truth reports from %s", len(mapping), path)
            return mapping
    except Exception as exc:  # noqa: BLE001
        logger.warning("Ground truth loading failed for %s: %s", path, exc)

    logger.info("No ground-truth reports found at %s; running without reference text.", path)
    return {}

# ---------------------------
# NORMALIZATION & CONTEXT
# ---------------------------

def build_bug_payload(row: Dict[str, str]) -> Dict[str, Any]:
    # Prefer 'description' column if you later add it. Else fallback to summary.
    description = (row.get("description") or "").strip()
    if not description:
        description = (row.get("summary") or "").strip()

    return {
        "bug_id": row.get("bug_id"),
        "type": row.get("classification", ""),
        "product": row.get("product", ""),
        "component": row.get("component", ""),
        "status": row.get("status", ""),
        "resolution": row.get("resolution", ""),
        "summary": (row.get("summary") or "").strip(),
        "description": description,
        "severity": row.get("severity", ""),
        "priority": row.get("priority", ""),
        "url": row.get("url", ""),
        "creation_time": row.get("creation_time", ""),
    }


# ---------------------------
# PIPELINE
# ---------------------------

def prepare_evaluation_entries(
    bug_rows: Iterable[Dict[str, str]],
    embedder: BugReportEmbedder,
    retriever: BugReportRetriever,
    generator: BugReportGenerator,
    ground_truth_map: Optional[Dict[str, str]] = None,
    sleep_between_calls: float = 0.0,
    min_ref_tokens: int = 40,
    top_k_contexts: Optional[int] = None,
) -> List[Dict[str, Any]]:
    if top_k_contexts is None:
        top_k_contexts = 8  # a bit richer by default

    entries: List[Dict[str, Any]] = []
    for idx, row in enumerate(bug_rows, start=1):
        bug_id = str(row.get("bug_id") or "").strip()
        question = (row.get("summary") or "").strip()
        if not question:
            logger.debug("Skipping bug %s due to missing summary question.", bug_id)
            continue

        bug_payload = build_bug_payload(row)

        # 1) Embed query
        try:
            query_embedding = embedder.embed_text(question)
        except Exception as exc:  # noqa: BLE001
            logger.error("Embedding failed for bug %s: %s", bug_id, exc)
            continue

        # 2) Vector retrieval with filter relax
        try:
            similar_candidates = retriever.retrieve_similar_bugs(
                query_embedding=query_embedding,
                bug_type=bug_payload.get("type", ""),
                product=bug_payload.get("product", ""),
                component=bug_payload.get("component", ""),
                top_k=max(top_k_contexts, Config.TOP_K_RESULTS),
                return_scores=True,
            )
        except Exception as exc:
            logger.error("Retrieval failed for bug %s: %s", bug_id, exc)
            continue

        candidate_ids = [
            str(item.get("bug_id") or "").strip()
            for item in (similar_candidates or [])
            if item and str(item.get("bug_id") or "").strip()
        ]

        if not candidate_ids:
            logger.debug("Skipping bug %s due to missing candidate IDs.", bug_id)
            continue

        try:
            candidate_details = retriever.get_bug_details(candidate_ids)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to fetch details for bug %s candidates: %s", bug_id, exc)
            candidate_details = []

        detail_map = {
            str(item.get("bug_id") or "").strip(): item for item in (candidate_details or [])
        }

        context_summaries: List[str] = []
        for candidate_id in candidate_ids[:top_k_contexts]:
            metadata = detail_map.get(candidate_id, {})
            summary = (
                metadata.get("summary")
                or metadata.get("title")
                or metadata.get("description")
                or metadata.get("text")
                or ""
            ).strip()
            if summary:
                context_summaries.append(f"Bug {candidate_id}: {summary}")
            else:
                context_summaries.append(f"Bug {candidate_id}: (no summary available)")

        reference_text = "\n".join(context_summaries)

        reports_dir = Config.OUTPUTS_DIR / "evaluation" / "reports"
        # Build the full report file using the report generator and read it as the answer
        try:
            result = generate_report(bug_payload, candidate_ids, output_dir=reports_dir)
            # generate_report now returns (report_path, prompt)
            if isinstance(result, tuple) or isinstance(result, list):
                report_path, prompt_messages = result[0], result[1]
            else:
                report_path, prompt_messages = result, None
            answer_text = Path(report_path).read_text(encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to build/read report for bug %s: %s", bug_id, exc)
            answer_text = ""
            prompt_messages = None

        # Prefer ground-truth reports when available (matching bug id)
        reference = ""
        if ground_truth_map and str(bug_id) in ground_truth_map:
            reference = ground_truth_map.get(str(bug_id), "")

        # Use the prompt messages (system + user content) as the 'question'
        # for evaluation if available; otherwise fall back to the bug summary.
        if prompt_messages and isinstance(prompt_messages, list):
            # join system+user messages into a single string to serve as the prompt
            combined_prompt = "\n\n".join([m.get("content", "") for m in prompt_messages if m.get("content")])
        else:
            combined_prompt = question

        entries.append(
            {
                "bug_id": bug_id,
                "question": (combined_prompt or question).strip(),
                "answer": (answer_text or "").strip(),
                "contexts": context_summaries or [""],
                "reference": reference.strip(),
            }
        )

        logger.info(
            "Prepared evaluation entry %d for bug %s (contexts=%d)",
            idx,
            bug_id,
            len(context_summaries),
        )
        if sleep_between_calls > 0:
            time.sleep(sleep_between_calls)

    return entries

# ---------------------------
# EVALUATION & SAVING
# ---------------------------

def _rename_metric(metric_name: str) -> str:
    return "answer_relevance" if metric_name == "answer_relevancy" else metric_name


def run_ragas_evaluation(entries: List[Dict[str, Any]]) -> Tuple[Dict[str, float], List[Dict[str, Any]], Dict[str, float]]:
    if not entries:
        raise ValueError("No hay datos para evaluar. Verifica ground truth y bugs disponibles.")

    dataset = Dataset.from_list(
        [
            {
                "question": item["question"],
                "answer": item["answer"],
                "contexts": item["contexts"],
                "reference": item["reference"],
            }
            for item in entries
        ]
    )

    # Use a set of metrics that includes reference-based and context-based
    # measurements. `answer_relevancy` is a reference-based metric provided by
    # ragas; normalize its name later to `answer_relevance` for output.
    metrics = [context_precision, faithfulness, answer_relevancy, context_recall]
    evaluation_result = evaluate(dataset=dataset, metrics=metrics)

    overall_scores: Dict[str, float] = {}
    non_zero_scores: Dict[str, float] = {}
    per_sample_rows: List[Dict[str, Any]] = []

    metric_names = list(evaluation_result.scores[0].keys()) if evaluation_result.scores else []
    # Compute overall and non-zero averages per metric
    for metric_name in metric_names:
        values = evaluation_result[metric_name]
        numeric_vals = [
            float(v)
            for v in values
            if isinstance(v, (int, float)) and not math.isnan(float(v))
        ]
        normalized = _rename_metric(metric_name)
        overall_scores[normalized] = (sum(numeric_vals) / len(numeric_vals)) if numeric_vals else float("nan")

        non_zero_vals = [val for val in numeric_vals if val != 0.0]
        non_zero_scores[normalized] = (sum(non_zero_vals) / len(non_zero_vals)) if non_zero_vals else float("nan")

    # Build per-sample rows that only contain the bug_id and the metric scores
    if evaluation_result.scores:
        for entry, scores in zip(entries, evaluation_result.scores):
            row = {"bug_id": entry.get("bug_id")}
            for metric_name in metric_names:
                row[_rename_metric(metric_name)] = scores.get(metric_name)
            per_sample_rows.append(row)

    return overall_scores, per_sample_rows, non_zero_scores

# ---------------------------
# CLI & MAIN
# ---------------------------

def save_results(
    overall_scores: Dict[str, float],
    per_sample_rows: List[Dict[str, Any]],
    output_dir: Optional[Path],
    entries: Optional[List[Dict[str, Any]]] = None,
) -> None:
    if not output_dir:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "ragas_summary.csv"
    per_sample_path = output_dir / "ragas_per_sample.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "score"])
        for name, value in overall_scores.items():
            writer.writerow([name, f"{value:.4f}"])
    if per_sample_rows:
        fieldnames = sorted({key for row in per_sample_rows for key in row.keys()})
        with per_sample_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in per_sample_rows:
                writer.writerow(row)
        # Write per-bug detailed files for inspection (scores + question/answer/reference/contexts)
        details_dir = output_dir / "details"
        details_dir.mkdir(parents=True, exist_ok=True)
        if entries:
            for row, entry in zip(per_sample_rows, entries):
                bid = str(entry.get("bug_id") or row.get("bug_id") or "unknown")
                safe_name = f"bug_{bid}_detail.txt"
                detail_path = details_dir / safe_name
                try:
                    with detail_path.open("w", encoding="utf-8") as fh:
                        fh.write("Scores:\n")
                        for k, v in sorted(row.items()):
                            if k == "bug_id":
                                continue
                            fh.write(f"{k}: {v}\n")
                        fh.write("\n")
                        fh.write("Question:\n")
                        fh.write((entry.get("question") or "") + "\n\n")
                        fh.write("Answer:\n")
                        fh.write((entry.get("answer") or "") + "\n\n")
                        fh.write("Reference:\n")
                        fh.write((entry.get("reference") or "") + "\n\n")
                        fh.write("Contexts:\n")
                        # contexts may be list or string
                        contexts = entry.get("contexts") or []
                        if isinstance(contexts, (list, tuple)):
                            fh.write("\n".join(contexts))
                        else:
                            fh.write(str(contexts))
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Failed to write detail file for %s: %s", bid, exc)
    logger.info("Saved summary to %s and per-sample scores to %s", summary_path, per_sample_path)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluar la calidad del RAG usando ragas (robusto y extractivo).")
    parser.add_argument("--bugs-csv", type=Path, default=Config.DATA_DIR / "bugs_since.csv", help="CSV con bugs.")
    parser.add_argument("--comments-csv", type=Path, default=Config.DATA_DIR / "bugs_comments.csv", help="CSV con comentarios (ground truth).")
    parser.add_argument("--limit", type=int, default=None, help="Cantidad máxima de bugs a evaluar.")
    parser.add_argument("--sleep", type=float, default=0.0, help="Pausa en segundos entre llamadas al LLM.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Nivel de logging.")
    parser.add_argument("--min-ref-tokens", type=int, default=40, help="Mínimo de tokens del ground truth.")
    parser.add_argument("--top-k-contexts", type=int, default=None, help="Contextos re-rankeados a inyectar (default=8).")
    parser.add_argument(
        "--dataset",
        choices=["default", "test"],
        default="default",
        help="Seleccionar dataset preconfigurado (default usa data/bugs_since.csv, test usa data/test/test_bugs.csv).",
    )
    return parser


def main(args: Optional[List[str]] = None) -> None:
    parser = build_arg_parser()
    cli_args = parser.parse_args(args=args)

    logging.basicConfig(
        level=getattr(logging, cli_args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    Config.validate_config()

    bugs_csv_path = cli_args.bugs_csv
    comments_csv_path = cli_args.comments_csv

    if cli_args.dataset == "test":
        bugs_csv_path = TEST_BUGS_CSV
        comments_csv_path = TEST_COMMENTS_CSV
        logger.info(
            "Using test dataset: bugs=%s comments=%s",
            bugs_csv_path,
            comments_csv_path,
        )

    bug_rows = load_bug_rows(bugs_csv_path, limit=cli_args.limit)
    # Load ground truth mapping. For the `test` dataset we expect a directory
    # with per-bug triage reports under data/test/ground_truths
    ground_truth_map = {}
    if cli_args.dataset == "test":
        gt_dir = TEST_DATA_DIR / "ground_truths"
        ground_truth_map = load_ground_truth(gt_dir)
    else:
        # If a comments CSV is explicitly provided, keep previous behaviour (no mapping)
        ground_truth_map = load_ground_truth(comments_csv_path) if comments_csv_path.exists() else {}

    embedder = BugReportEmbedder()
    retriever = BugReportRetriever()
    generator = BugReportGenerator(retriever=retriever)

    entries = prepare_evaluation_entries(
        bug_rows=bug_rows,
        embedder=embedder,
        retriever=retriever,
        generator=generator,
        ground_truth_map=ground_truth_map,
        sleep_between_calls=cli_args.sleep,
        min_ref_tokens=cli_args.min_ref_tokens,
        top_k_contexts=cli_args.top_k_contexts,
    )

    overall_scores, per_sample_rows, non_zero_scores = run_ragas_evaluation(entries)
    logger.info("Overall ragas scores (all samples): %s", overall_scores)
    logger.info("Overall ragas scores (excluding 0.0 values): %s", non_zero_scores)

    save_results(
        overall_scores=overall_scores,
        per_sample_rows=per_sample_rows,
        output_dir=Config.OUTPUTS_DIR / "evaluation",
        entries=entries,
    )


if __name__ == "__main__":
    main()
