"""
Este archivo va a recibir el nuevo bug + contexto retrieved == genera respuesta del LLM
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from openai import OpenAI

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

        return str(report_path), prompt

    def _fetch_context(self, similar_bug_ids: List[str]) -> List[Dict[str, Any]]:
        """Retrieve metadata for similar bugs from the original CSV data."""
        if not similar_bug_ids:
            logger.warning("No similar bug IDs provided; skipping context assembly")
            return []

        try:
            csv_path = Config.PROJECT_ROOT / "data" / "bugs_since.csv"
            if not csv_path.exists():
                logger.error(f"CSV data file not found: {csv_path}")
                return []
            
            logger.info(f"Loading bug data from CSV: {csv_path}")
            df = pd.read_csv(csv_path)

            normalized_bug_ids = []
            for bug_id in similar_bug_ids:
                try:
                    normalized_id = int(float(bug_id))
                    normalized_bug_ids.append(normalized_id)
                except (ValueError, TypeError):
                    logger.warning(f"Could not convert bug_id '{bug_id}' to integer")
                    continue
            
            logger.info(f"Looking for bugs with IDs: {normalized_bug_ids}")

            matching_bugs = df[df['bug_id'].isin(normalized_bug_ids)]
            logger.info(f"Found {len(matching_bugs)} matching bugs in CSV")
   
            context = []
            for _, row in matching_bugs.iterrows():
                bug_dict = {
                    'bug_id': str(int(row['bug_id'])),
                    'summary': row.get('summary', ''),
                    'product': row.get('product', ''),
                    'component': row.get('component', ''),
                    'status': row.get('status', ''),
                    'resolution': row.get('resolution', ''),
                    'priority': row.get('priority', ''),
                    'severity': row.get('severity', ''),
                    'classification': row.get('classification', ''),
                    'platform': row.get('platform', ''),
                    'version': row.get('version', ''),
                    'creation_time': row.get('creation_time', ''),
                    'url': row.get('url', ''),
                    'blocks': row.get('blocks', ''),
                }
                context.append(bug_dict)

            context = self._enrich_with_comments(context)

            context = self._enrich_with_blocked_bugs(context, df)
            
            logger.info(f"Successfully fetched {len(context)} contextual bugs for report generation")
            return context
            
        except Exception as exc:
            logger.error(f"Failed to fetch bug context from CSV: {exc}")
            return []

    def _enrich_with_comments(self, context: List[Dict[str, Any]], max_comments_per_bug: int = 5) -> List[Dict[str, Any]]:
        """
        Fetch comments for each bug from bugs_comments.csv
        
        Args:
            context: List of bug dictionaries
            max_comments_per_bug: Maximum number of comments to fetch per bug
            
        Returns:
            Enhanced context with comments added
        """
        try:
            comments_csv_path = Config.PROJECT_ROOT / "data" / "bugs_comments.csv"
            if not comments_csv_path.exists():
                logger.warning(f"Comments CSV file not found: {comments_csv_path}")
                return context
            
            logger.info(f"Loading comments data from CSV: {comments_csv_path}")
            comments_df = pd.read_csv(comments_csv_path)

            bug_ids = [int(bug['bug_id']) for bug in context]
            logger.info(f"Fetching comments for {len(bug_ids)} bugs")

            relevant_comments = comments_df[comments_df['bug_id'].isin(bug_ids)]
 
            for bug in context:
                bug_id = int(bug['bug_id'])
                bug_comments = relevant_comments[relevant_comments['bug_id'] == bug_id]

                bug_comments = bug_comments.sort_values('creation_time', ascending=False).head(max_comments_per_bug)

                comments_list = []
                for _, comment_row in bug_comments.iterrows():
                    comment_dict = {
                        'comment_id': comment_row.get('comment_id', ''),
                        'creation_time': comment_row.get('creation_time', ''),
                        'text': comment_row.get('text', '')[:500]  
                    }
                    comments_list.append(comment_dict)
                
                bug['comments'] = comments_list
                logger.debug(f"Added {len(comments_list)} comments to bug {bug_id}")
            
            total_comments = sum(len(bug.get('comments', [])) for bug in context)
            logger.info(f"Successfully fetched {total_comments} comments across {len(context)} bugs")
            
        except Exception as exc:
            logger.error(f"Failed to fetch comments: {exc}")
            for bug in context:
                bug['comments'] = []
        
        return context

    def _enrich_with_blocked_bugs(self, context: List[Dict[str, Any]], main_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Fetch information about bugs that are blocked by each context bug
        
        Args:
            context: List of bug dictionaries
            main_df: The main bugs DataFrame
            
        Returns:
            Enhanced context with blocked bugs information
        """
        try:
            logger.info("Enriching context with blocked bugs information")
            
            for bug in context:
                blocks_field = bug.get('blocks', '')
                if not blocks_field or pd.isna(blocks_field):
                    bug['blocked_bugs'] = []
                    continue
                try:
                    blocked_ids = [int(bid.strip()) for bid in str(blocks_field).split(',') if bid.strip().isdigit()]
                    
                    if not blocked_ids:
                        bug['blocked_bugs'] = []
                        continue

                    blocked_bugs_info = []
                    for blocked_id in blocked_ids[:5]:
                        blocked_bug_row = main_df[main_df['bug_id'] == blocked_id]
                        if not blocked_bug_row.empty:
                            blocked_info = {
                                'bug_id': str(blocked_id),
                                'summary': blocked_bug_row.iloc[0].get('summary', 'N/A'),
                                'status': blocked_bug_row.iloc[0].get('status', 'N/A'),
                                'priority': blocked_bug_row.iloc[0].get('priority', 'N/A')
                            }
                            blocked_bugs_info.append(blocked_info)
                    
                    bug['blocked_bugs'] = blocked_bugs_info
                    if blocked_bugs_info:
                        logger.debug(f"Bug {bug['bug_id']} blocks {len(blocked_bugs_info)} other bugs")
                    
                except (ValueError, AttributeError) as e:
                    logger.debug(f"Could not parse blocks field for bug {bug['bug_id']}: {blocks_field}")
                    bug['blocked_bugs'] = []
            
            total_blocked = sum(len(bug.get('blocked_bugs', [])) for bug in context)
            logger.info(f"Found {total_blocked} blocked bug relationships")
            
        except Exception as exc:
            logger.error(f"Failed to enrich with blocked bugs: {exc}")
            for bug in context:
                bug['blocked_bugs'] = []
        
        return context

    def _build_prompt(
        self,
        bug_data: Dict[str, Any],
        contextual_bugs: List[Dict[str, Any]],
    ) -> List[Dict[str, str]]:
        """
        Build a chat-style prompt for the LLM using bug details and retrieved context.
        """
        system_message = (
            "You are an expert bug triage assistant that analyzes new software bug reports. "
            "Given a new bug report and context from similar historical bugs (including their comments and relationships), provide:\n\n"
            "1. **FIELD PREDICTIONS**: Based on similar bugs, predict appropriate values for:\n"
            "   - Priority (P1=Critical, P2=High, P3=Normal, P4=Low, P5=Lowest)\n"
            "   - Severity (S1=Blocker, S2=Critical, S3=Major, S4=Normal, N/A=Not classified)\n"
            "   - Platform (if not specified by user. Options: ARM, ARM64, All, Desktop, Other, RISCV64, Unspecified, x86, x86_64)\n\n"
            "2. **ROOT CAUSE ANALYSIS**: Potential technical causes and related modules based on:\n"
            "   - Similar bug patterns and their resolutions\n"
            "   - Comments from similar bugs that reveal technical insights\n"
            "   - Bug relationships (what bugs block or are blocked by similar issues)\n\n"
            "3. **TRIAGE RECOMMENDATIONS**: Suggested next steps for investigation and resolution\n\n"
            "**CONTEXT AVAILABLE**: For each similar bug, you have access to:\n"
            "- Basic metadata (priority, severity, status, etc.)\n"
            "- Recent comments from developers/users that may contain technical details\n"
            "- Information about related bugs that this bug blocks or is blocked by\n\n"
            "**FIELD INFERENCE RULES (use contexts first)**\n"
            "- Derive Priority/Severity by MAJORITY VOTE over retrieved similar bugs:\n"
            "  - Compute the most frequent value among context items for each field.\n"
            "  - If a value appears most frequently among similar bugs, USE THAT VALUE.\n"
            "  - IGNORE invalid values like '--', '---', 'unknown', null, empty strings when counting.\n"
            "  - Break ties by preferring: (1) same component/product, (2) most recent, (3) higher evidence density (more technical comments).\n"
            "- If no clear majority (after ignoring invalid values):\n"
            "  - If the new bug's summary contains 'Intermittent' or 'single tracking bug' or the component is Talos/mochitest/reftest/APZ, set Priority=P5 and Severity=S4.\n"
            "  - For crashes/security: prefer P1-P2, S1-S2.\n"
            "  - Otherwise, use the most conservative valid value from similar bugs (prefer higher P numbers, higher S numbers).\n"
            "- Platform: use most common valid platform from similar bugs; if tied, use 'Unspecified'.\n"
            "- Always cite the context IDs supporting your choices, e.g., 'Priority=P5 (C1,C3,C5)'.\n"
            "- Make educated predictions based on similar bugs. Avoid defaulting to 'Unknown'.\n\n"
            "Use this rich context to make informed predictions and provide detailed technical analysis."
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
        status = bug_data.get("status", "NEW")
        resolution = bug_data.get("resolution", "")

        version = bug_data.get("version", "")
        version_text = f" (Version: {version})" if version else ""

        platform = bug_data.get("platform", "")
        platform_text = f"\nPlatform: {platform}" if platform else ""

        return (
            f"Bug ID: {bug_id}\n"
            f"Product / Component: {product} / {component}{version_text}\n"
            f"Status / Resolution: {status} / {resolution}\n"
            f"Summary: {summary}{platform_text}\n"
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
                f"Product/Component: {bug.get('product', 'N/A')} / {bug.get('component', 'N/A')}",
                f"Priority: {bug.get('priority', 'N/A')} | Severity: {bug.get('severity', 'N/A')}",
                f"Classification: {bug.get('classification', 'N/A')} | Platform: {bug.get('platform', 'N/A')}",
                f"Status: {bug.get('status', 'unknown')} | Resolution: {bug.get('resolution', 'unknown')}",
            ]

            version = bug.get('version')
            if version and version != 'unspecified':
                block.append(f"Version: {version}")

            comments = bug.get('comments', [])
            if comments:
                block.append(f"Recent Comments ({len(comments)}):")
                for i, comment in enumerate(comments[:3], 1): 
                    comment_text = comment.get('text', '')[:200]  
                    comment_time = comment.get('creation_time', 'unknown')
                    block.append(f"  Comment {i} ({comment_time[:10]}): {comment_text}{'...' if len(comment.get('text', '')) > 200 else ''}")

            blocked_bugs = bug.get('blocked_bugs', [])
            if blocked_bugs:
                block.append(f"Blocks {len(blocked_bugs)} bugs:")
                for blocked in blocked_bugs[:3]: 
                    block.append(f"  -> Bug {blocked.get('bug_id', 'N/A')}: {blocked.get('summary', 'N/A')[:100]}{'...' if len(blocked.get('summary', '')) > 100 else ''} [{blocked.get('status', 'N/A')}]")

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
