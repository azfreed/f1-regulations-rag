"""Command-line interface.

The CLI is intentionally thin: it parses flags, builds a :class:`Settings`, and
delegates to the pipeline/registries. No implementation is hard-coded here - every
component is chosen by name and resolved through its registry, so new
implementations become available on the CLI simply by registering themselves.

Subcommands
-----------
- ``ingest``   : build a searchable index from the PDFs.
- ``ask``      : answer a question, printing a full trace.
- ``diagnose`` : produce the milestone per-PDF diagnostic reports.
- ``evaluate`` : run the eval dataset and save a comparable experiment record.
- ``components``: list available implementations for each stage.
"""

from __future__ import annotations

import argparse
import sys

from .config import Settings
from .logging_utils import configure_logging, get_logger

logger = get_logger(__name__)


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--log-level", default=None, help="DEBUG/INFO/WARNING/ERROR")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="f1_rag", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    # --- ingest ---
    p_ing = sub.add_parser("ingest", help="extract, chunk, embed, and index the PDFs")
    _add_common(p_ing)
    p_ing.add_argument("--chunker", default=None, help="regulation | fixed_window")
    p_ing.add_argument("--embedder", default=None, help="minilm | hashing")
    p_ing.add_argument("--index", default=None, help="numpy | chroma")
    p_ing.add_argument("--pdf", action="append", default=None, help="specific PDF filename(s)")
    p_ing.add_argument("--force", action="store_true", help="rebuild even if up-to-date")

    # --- ask ---
    p_ask = sub.add_parser("ask", help="answer a question over the regulations")
    _add_common(p_ask)
    p_ask.add_argument("question", help="the question to answer")
    p_ask.add_argument("--chunker", default=None)
    p_ask.add_argument("--embedder", default=None)
    p_ask.add_argument("--index", default=None)
    p_ask.add_argument("--retriever", default=None, help="vector | bm25 | hybrid")
    p_ask.add_argument("--reranker", default=None, help="none")
    p_ask.add_argument("--section", default=None, help="restrict to a section letter A-F")
    p_ask.add_argument("--k", type=int, default=None, help="number of candidates")
    p_ask.add_argument("--no-generate", action="store_true", help="retrieval + context only")
    p_ask.add_argument("--json-trace", action="store_true", help="also write a JSON trace")

    # --- diagnose ---
    p_diag = sub.add_parser("diagnose", help="produce per-PDF diagnostic reports")
    _add_common(p_diag)
    p_diag.add_argument("--pdf", action="append", default=None)
    p_diag.add_argument("--no-visuals", action="store_true", help="skip page rendering")

    # --- evaluate ---
    p_eval = sub.add_parser("evaluate", help="run the evaluation dataset")
    _add_common(p_eval)
    p_eval.add_argument("--chunker", default=None)
    p_eval.add_argument("--embedder", default=None)
    p_eval.add_argument("--index", default=None)
    p_eval.add_argument("--retriever", default=None)
    p_eval.add_argument("--reranker", default=None)
    p_eval.add_argument("--k", type=int, default=None)
    p_eval.add_argument("--dataset", default=None, help="path to a dataset JSON")
    p_eval.add_argument("--generate", action="store_true", help="also run generation")

    # --- components ---
    p_comp = sub.add_parser("components", help="list registered implementations")
    _add_common(p_comp)

    return parser


def _cmd_ingest(args: argparse.Namespace, settings: Settings) -> int:
    from .pipeline import ingest

    result = ingest(
        settings,
        chunker=args.chunker or settings.chunker,
        embedder=args.embedder or settings.embedder,
        index=args.index or settings.index,
        pdf_filenames=args.pdf,
        force=args.force,
    )
    status = "up-to-date (skipped)" if result.skipped else "built"
    print(
        f"Index {status}: {result.index_dir}\n"
        f"  documents: {result.n_documents}\n  chunks: {result.n_chunks}"
    )
    return 0


def _cmd_ask(args: argparse.Namespace, settings: Settings) -> int:
    from .pipeline import QueryEngine, render_trace
    from .tracing import write_trace_json

    engine = QueryEngine.load(
        settings, chunker=args.chunker, embedder=args.embedder, index=args.index
    )
    answer, trace = engine.answer(
        args.question,
        retriever=args.retriever,
        reranker=args.reranker,
        section=args.section,
        k=args.k,
        generate=not args.no_generate,
    )
    print(render_trace(trace))
    if args.json_trace:
        from datetime import datetime, timezone

        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = write_trace_json(trace, settings.traces_dir / f"{ts}.json")
        print(f"\n[trace JSON written to {path}]")
    return 0


def _cmd_diagnose(args: argparse.Namespace, settings: Settings) -> int:
    from .diagnostics import run_diagnostics

    written = run_diagnostics(
        settings, pdf_filenames=args.pdf, render_visuals=not args.no_visuals
    )
    print("Diagnostic reports written:")
    for p in written:
        print(f"  - {p}")
    return 0


def _cmd_evaluate(args: argparse.Namespace, settings: Settings) -> int:
    from .evaluation import load_dataset, run_evaluation, seed_dataset

    dataset = load_dataset(args.dataset) if args.dataset else seed_dataset()
    metrics, path = run_evaluation(
        settings,
        dataset,
        chunker=args.chunker,
        embedder=args.embedder,
        index=args.index,
        retriever=args.retriever,
        reranker=args.reranker,
        k=args.k,
        generate=args.generate,
    )
    print("Evaluation metrics:")
    for key, val in metrics.model_dump().items():
        print(f"  {key}: {val}")
    print(f"\nSaved to {path}")
    return 0


def _cmd_components(args: argparse.Namespace, settings: Settings) -> int:
    # Importing the packages triggers registration of all implementations.
    from . import chunking, context, embeddings, generation, indexing, reranking, retrieval

    print("Registered components:")
    print(f"  chunkers   : {', '.join(chunking.chunker_registry.available())}")
    print(f"  embedders  : {', '.join(embeddings.embedder_registry.available())}")
    print(f"  indexes    : {', '.join(indexing.index_registry.available())}")
    print(f"  retrievers : {', '.join(retrieval.retriever_registry.available())}")
    print(f"  rerankers  : {', '.join(reranking.reranker_registry.available())}")
    print(f"  context    : {', '.join(context.assembler_registry.available())}")
    print(f"  generators : {', '.join(generation.generator_registry.available())}")
    return 0


_COMMANDS = {
    "ingest": _cmd_ingest,
    "ask": _cmd_ask,
    "diagnose": _cmd_diagnose,
    "evaluate": _cmd_evaluate,
    "components": _cmd_components,
}


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(getattr(args, "log_level", None))
    settings = Settings()
    try:
        return _COMMANDS[args.command](args, settings)
    except Exception as exc:  # noqa: BLE001 - top-level: report cleanly, non-zero exit
        logger.error("%s: %s", type(exc).__name__, exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
