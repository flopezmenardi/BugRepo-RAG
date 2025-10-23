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


def load_ground_truth(comments_csv: Path) -> Dict[str, str]:
    comments_map: Dict[str, List[str]] = defaultdict(list)
    try:
        with comments_csv.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                bug_id = str(row.get("bug_id") or "").strip()
                text = (row.get("text") or "").strip()
                if bug_id and text:
                    comments_map[bug_id].append(text)
    except FileNotFoundError:
        logger.warning("Comments CSV not found at %s; evaluation may lack ground truth.", comments_csv)

    ground_truth = {
        bug_id: " ".join(chunks)
        for bug_id, chunks in comments_map.items()
        if chunks
    }
    logger.info("Collected ground truth for %d bugs", len(ground_truth))
    return ground_truth

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
    ground_truth_map: Dict[str, str],
    embedder: BugReportEmbedder,
    retriever: BugReportRetriever,
    generator: BugReportGenerator,
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
        ground_truth = ground_truth_map.get(bug_id)

        if not question or not ground_truth or len(ground_truth.split()) < min_ref_tokens:
            logger.debug("Skipping bug %s due to missing/short ground truth.", bug_id)
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
                classification=bug_payload.get("classification", ""),
                product=bug_payload.get("product", ""),
                component=bug_payload.get("component", ""),
                top_k=max(top_k_contexts, Config.TOP_K_RESULTS),
                return_scores=True,
            )
        except Exception as exc:
            logger.error("Retrieval failed for bug %s: %s", bug_id, exc)
            continue

        answer_text, _usage, contexts, _ranked_meta = generator.generate_contextual_answer(
            bug_data=bug_payload,
            candidates=similar_candidates,
            top_k=top_k_contexts,
            reference_text=ground_truth,
            query_text=question,
        )

        entries.append(
            {
                "bug_id": bug_id,
                "question": question,
                "answer": answer_text.strip(),
                "contexts": contexts or [""],
                "reference": ground_truth,
            }
        )

        logger.info("Prepared evaluation entry %d for bug %s (contexts=%d)", idx, bug_id, len(contexts))
        if sleep_between_calls > 0:
            time.sleep(sleep_between_calls)

    return entries

# ---------------------------
# EVALUATION & SAVING
# ---------------------------

def _rename_metric(metric_name: str) -> str:
    return "answer_relevance" if metric_name == "answer_relevancy" else metric_name


def run_ragas_evaluation(entries: List[Dict[str, Any]]) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
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

    metrics = [context_precision, context_recall, faithfulness, answer_relevancy]
    evaluation_result = evaluate(dataset=dataset, metrics=metrics)

    overall_scores: Dict[str, float] = {}
    per_sample_rows: List[Dict[str, Any]] = []

    metric_names = list(evaluation_result.scores[0].keys()) if evaluation_result.scores else []
    for metric_name in metric_names:
        values = evaluation_result[metric_name]
        numeric_vals = [
            float(v)
            for v in values
            if isinstance(v, (int, float)) and not math.isnan(float(v))
        ]
        normalized = _rename_metric(metric_name)
        overall_scores[normalized] = (sum(numeric_vals) / len(numeric_vals)) if numeric_vals else float("nan")

    try:
        df = evaluation_result.to_pandas()
        df = df.rename(columns={name: _rename_metric(name) for name in metric_names})
        per_sample_rows = df.to_dict(orient="records")
    except Exception as exc:
        logger.warning("Failed to convert evaluation results to DataFrame: %s", exc)
        for entry, scores in zip(entries, evaluation_result.scores):
            row = {"bug_id": entry["bug_id"]}
            for metric_name in metric_names:
                row[_rename_metric(metric_name)] = scores.get(metric_name)
            per_sample_rows.append(row)

    return overall_scores, per_sample_rows

# ---------------------------
# CLI & MAIN
# ---------------------------

def save_results(
    overall_scores: Dict[str, float],
    per_sample_rows: List[Dict[str, Any]],
    output_dir: Optional[Path],
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
    ground_truth_map = load_ground_truth(comments_csv_path)

    embedder = BugReportEmbedder()
    retriever = BugReportRetriever()
    generator = BugReportGenerator(retriever=retriever)

    entries = prepare_evaluation_entries(
        bug_rows=bug_rows,
        ground_truth_map=ground_truth_map,
        embedder=embedder,
        retriever=retriever,
        generator=generator,
        sleep_between_calls=cli_args.sleep,
        min_ref_tokens=cli_args.min_ref_tokens,
        top_k_contexts=cli_args.top_k_contexts,
    )

    overall_scores, per_sample_rows = run_ragas_evaluation(entries)
    logger.info("Overall ragas scores: %s", overall_scores)

    save_results(
        overall_scores=overall_scores,
        per_sample_rows=per_sample_rows,
        output_dir=Config.OUTPUTS_DIR / "evaluation",
    )


if __name__ == "__main__":
    main()
