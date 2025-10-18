"""
Este archivo va a recibir el nuevo bug + contexto retrieved == genera respuesta del LLM
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from openai import OpenAI

# Ensure project root is on the path for absolute imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config import Config
from src.retrieval.retriever import BugReportRetriever

logger = logging.getLogger(__name__)


class BugReportGenerator:
    """
    Generates an LLM-based report for a new bug leveraging retrieved context.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        retriever: Optional[BugReportRetriever] = None,
    ):
        Config.validate_config()

        self.client = OpenAI(api_key=Config.OPENAI_API_KEY)
        self.model_name = model_name or Config.LLM_MODEL
        self.temperature = (
            Config.LLM_TEMPERATURE if temperature is None else float(temperature)
        )
        self.max_tokens = (
            Config.MAX_TOKENS if max_tokens is None else int(max_tokens)
        )
        self.retriever = retriever

        logger.info(
            "Initialized BugReportGenerator with model=%s, temperature=%.2f, max_tokens=%d",
            self.model_name,
            self.temperature,
            self.max_tokens,
        )

    def generate_report(
        self,
        bug_data: Dict[str, Any],
        similar_bug_ids: List[str],
        output_dir: Optional[Path] = None,
    ) -> str:
        """
        Generate a final report for the input bug.

        Args:
            bug_data: Original bug metadata/content.
            similar_bug_ids: Identifiers returned by the retriever step.
            output_dir: Optional override for the output directory.

        Returns:
            Path to the generated report file (as string).
        """
        if not bug_data:
            raise ValueError("bug_data is required to generate a report")

        contextual_bugs = self._fetch_context(similar_bug_ids)
        prompt = self._build_prompt(bug_data, contextual_bugs)
        llm_text, usage = self._call_llm(prompt)
        report_path = self._write_report(
            bug_data,
            contextual_bugs,
            llm_text,
            usage,
            output_dir=output_dir,
        )

        return str(report_path)

    def _fetch_context(self, similar_bug_ids: List[str]) -> List[Dict[str, Any]]:
        """Retrieve metadata for similar bugs from the original CSV data."""
        if not similar_bug_ids:
            logger.warning("No similar bug IDs provided; skipping context assembly")
            return []

        try:
            # Load the original CSV data to get complete bug information
            csv_path = Config.PROJECT_ROOT / "data" / "sample_bugs.csv"
            if not csv_path.exists():
                logger.error(f"CSV data file not found: {csv_path}")
                return []
            
            logger.info(f"Loading bug data from CSV: {csv_path}")
            df = pd.read_csv(csv_path)
            
            # Convert bug IDs to integers for matching
            normalized_bug_ids = []
            for bug_id in similar_bug_ids:
                try:
                    # Convert float strings like "1546498.0" to integers
                    normalized_id = int(float(bug_id))
                    normalized_bug_ids.append(normalized_id)
                except (ValueError, TypeError):
                    logger.warning(f"Could not convert bug_id '{bug_id}' to integer")
                    continue
            
            logger.info(f"Looking for bugs with IDs: {normalized_bug_ids}")
            
            # Filter dataframe to get matching bugs
            matching_bugs = df[df['Bug ID'].isin(normalized_bug_ids)]
            logger.info(f"Found {len(matching_bugs)} matching bugs in CSV")
            
            # Convert to list of dictionaries
            context = []
            for _, row in matching_bugs.iterrows():
                bug_dict = {
                    'bug_id': str(int(row['Bug ID'])),
                    'summary': row.get('Summary', ''),
                    'type': row.get('Type', ''),
                    'product': row.get('Product', ''),
                    'component': row.get('Component', ''),
                    'status': row.get('Status', ''),
                    'resolution': row.get('Resolution', ''),
                    'updated': row.get('Updated', ''),
                    'assignee': row.get('Assignee', ''),
                }
                context.append(bug_dict)
            
            logger.info(f"Successfully fetched {len(context)} contextual bugs for report generation")
            return context
            
        except Exception as exc:
            logger.error(f"Failed to fetch bug context from CSV: {exc}")
            return []

    def _build_prompt(
        self,
        bug_data: Dict[str, Any],
        contextual_bugs: List[Dict[str, Any]],
    ) -> List[Dict[str, str]]:
        """
        Build a chat-style prompt for the LLM using bug details and retrieved context.
        """
        system_message = (
            "You are an assistant that analyzes software bug reports. "
            "Given a new bug and context from similar historical bugs, provide:\n"
            "1. A concise summary of the new bug\n"
            "2. Potential root causes or related modules based on context\n"
            "3. Suggested next steps for triage or resolution\n"
            "Keep the response actionable and reference similar bugs when useful."
        )

        bug_overview = self._format_bug_overview(bug_data)
        context_section = self._format_contextual_bugs(contextual_bugs)

        user_message = (
            "NEW BUG SUMMARY:\n"
            f"{bug_overview}\n\n"
            "RETRIEVED CONTEXT:\n"
            f"{context_section if context_section else 'No similar bugs available.'}\n\n"
            "Please craft the report now."
        )

        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]

    def _format_bug_overview(self, bug_data: Dict[str, Any]) -> str:
        """Format the incoming bug data into a readable block for the prompt."""
        summary = bug_data.get("summary") or bug_data.get("title") or "N/A"
        description = bug_data.get("description") or "No description provided."
        product = bug_data.get("product", "unknown product")
        component = bug_data.get("component", "unknown component")
        bug_id = bug_data.get("bug_id", "unknown")
        status = bug_data.get("status", "untracked")
        resolution = bug_data.get("resolution", "unresolved")

        return (
            f"Bug ID: {bug_id}\n"
            f"Product / Component: {product} / {component}\n"
            f"Status / Resolution: {status} / {resolution}\n"
            f"Summary: {summary}\n"
            f"Description: {description}"
        )

    def _format_contextual_bugs(self, contextual_bugs: List[Dict[str, Any]]) -> str:
        """Serialize similar bugs into a concise, token-friendly string."""
        if not contextual_bugs:
            return ""

        blocks: List[str] = []
        for idx, bug in enumerate(contextual_bugs, start=1):
            block = [
                f"[{idx}] Bug ID: {bug.get('bug_id', 'unknown')}",
                f"Summary: {bug.get('summary', 'N/A')}",
                f"Status: {bug.get('status', 'unknown')} | Resolution: {bug.get('resolution', 'unknown')}",
            ]

            description = bug.get("description")
            if description:
                block.append(f"Description: {description[:600]}{'...' if len(description) > 600 else ''}")

            blocks.append("\n".join(block))

        return "\n\n".join(blocks)

    def _call_llm(self, messages: List[Dict[str, str]]) -> Tuple[str, Dict[str, Any]]:
        """
        Execute the LLM call and return the generated text with token usage.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            choice = response.choices[0] if response.choices else None
            llm_text = ""

            if choice and choice.message and choice.message.content:
                llm_text = choice.message.content.strip()

            usage = self._normalize_usage(getattr(response, "usage", None))

            if not llm_text:
                logger.warning("LLM returned empty response; falling back to placeholder text")
                llm_text = "Unable to generate report. Please review the bug details manually."

            logger.debug("LLM usage stats: %s", usage)
            return llm_text, usage

        except Exception as exc:
            logger.error("LLM generation failed: %s", exc)
            return (
                "LLM call failed. Include bug summary and retrieved context when requesting manual review.",
                {},
            )

    def _write_report(
        self,
        bug_data: Dict[str, Any],
        contextual_bugs: List[Dict[str, Any]],
        llm_text: str,
        usage: Dict[str, Any],
        output_dir: Optional[Path] = None,
    ) -> Path:
        """
        Persist the final report to disk and return the path.
        """
        output_root = output_dir or Config.OUTPUTS_DIR
        output_root.mkdir(exist_ok=True, parents=True)

        bug_id = bug_data.get("bug_id", "unknown")
        report_path = output_root / f"bug_report_{bug_id}.txt"

        with report_path.open("w", encoding="utf-8") as handle:
            handle.write("Bug Report Analysis\n")
            handle.write("====================\n\n")
            handle.write(self._format_bug_overview(bug_data))
            handle.write("\n\n")
            handle.write("Similar Bugs Context\n")
            handle.write("--------------------\n")
            handle.write(
                f"{self._format_contextual_bugs(contextual_bugs) or 'No similar bugs could be retrieved.'}\n\n"
            )
            handle.write("LLM Generated Guidance\n")
            handle.write("----------------------\n")
            handle.write(f"{llm_text}\n\n")

            if usage:
                handle.write("Usage Metadata\n")
                handle.write("--------------\n")
                for key, value in usage.items():
                    handle.write(f"{key}: {value}\n")

        logger.info("Report written to %s", report_path)
        return report_path

    @staticmethod
    def _normalize_usage(usage_obj: Optional[Any]) -> Dict[str, Any]:
        """Convert OpenAI usage payloads into plain dictionaries for persistence."""
        if not usage_obj:
            return {}

        if isinstance(usage_obj, dict):
            return usage_obj

        if hasattr(usage_obj, "model_dump"):
            try:
                return usage_obj.model_dump()
            except Exception:
                pass

        usage_dict: Dict[str, Any] = {}
        for attr in ("prompt_tokens", "completion_tokens", "total_tokens"):
            if hasattr(usage_obj, attr):
                usage_dict[attr] = getattr(usage_obj, attr)

        return usage_dict


def generate_report(
    bug_data: Dict[str, Any],
    similar_bug_ids: List[str],
    output_dir: Optional[Path] = None,
) -> str:
    """
    Convenience function used by the pipeline to build the final report.
    """
    generator = BugReportGenerator()
    return generator.generate_report(bug_data, similar_bug_ids, output_dir=output_dir)
