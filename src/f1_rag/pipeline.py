"""Pipeline orchestration: wire stages together for ingestion and querying.

This module is the one place that composes the individually-testable stages into
end-to-end flows. It deliberately contains *no* stage logic itself - it only
selects implementations via the registries and passes typed models between them.

Two flows:
- :func:`ingest` : PDFs -> raw -> visuals -> clean -> parse -> chunk -> embed ->
  index (persisted, idempotent).
- :class:`QueryEngine` : load a persisted index and answer questions, recording a
  full :class:`QueryTrace` at every step.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from .chunking import chunker_registry
from .config import Settings
from .context import assembler_registry
from .embeddings import embedder_registry
from .errors import ConfigurationError, F1RagError
from .extraction.cleanup import clean_document
from .extraction.text import extract_document
from .extraction.visuals import extract_visuals
from .generation import generator_registry
from .indexing import index_registry
from .logging_utils import get_logger
from .models import (
    Answer,
    Chunk,
    RegulationSection,
    SourceDocumentMeta,
    read_jsonl,
    write_jsonl,
)
from .parsing.regulations import parse_articles
from .reranking import reranker_registry
from .retrieval import RetrievalDeps, normalize_query, retriever_registry
from .tracing import QueryTrace, TraceRecorder, render_trace_text

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Index identity + persistence layout
# ---------------------------------------------------------------------------
def index_key(index: str, chunker: str, embedder: str) -> str:
    """Directory name encoding the configuration, so configs don't clobber each other."""

    return f"{index}__{chunker}__{embedder}"


def _corpus_hash(pdf_paths: list[Path]) -> str:
    """Stable hash of the source corpus (name + size) for idempotency checks."""

    h = hashlib.sha1()
    for p in sorted(pdf_paths, key=lambda x: x.name):
        h.update(p.name.encode())
        h.update(str(p.stat().st_size).encode())
    return h.hexdigest()[:16]


def _build_embedder(settings: Settings, name: str):
    if name == "minilm":
        return embedder_registry.create(name, settings.embedding_model)
    return embedder_registry.create(name)


def _build_store(settings: Settings, name: str, index_dir: Path):
    if name == "chroma":
        return index_registry.create(name, persist_path=str(index_dir / "chroma"))
    return index_registry.create(name)


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------
@dataclass
class IngestResult:
    index_dir: Path
    n_chunks: int
    n_documents: int
    skipped: bool


def ingest(
    settings: Settings,
    chunker: str,
    embedder: str,
    index: str,
    pdf_filenames: list[str] | None = None,
    force: bool = False,
) -> IngestResult:
    """Run the full ingestion pipeline and persist a searchable index."""

    settings.ensure_dirs()
    raw_dir = settings.raw_dir
    pdfs = (
        [raw_dir / fn for fn in pdf_filenames]
        if pdf_filenames
        else sorted(raw_dir.glob("*.pdf"))
    )
    if not pdfs:
        raise ConfigurationError(f"no PDFs found in {raw_dir}")

    idx_dir = settings.indexes_dir / index_key(index, chunker, embedder)
    manifest_path = idx_dir / "manifest.json"
    corpus_hash = _corpus_hash(pdfs)

    # Idempotency: skip if an index for this exact corpus + config already exists.
    if manifest_path.exists() and not force:
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("corpus_hash") == corpus_hash:
            logger.info("index up-to-date (corpus unchanged); skipping. Use --force to rebuild.")
            return IngestResult(
                index_dir=idx_dir,
                n_chunks=manifest.get("n_chunks", 0),
                n_documents=manifest.get("n_documents", 0),
                skipped=True,
            )

    chunker_impl = chunker_registry.create(
        chunker, max_tokens=settings.chunk_max_tokens, overlap_tokens=settings.chunk_overlap_tokens
    )
    embedder_impl = _build_embedder(settings, embedder)

    all_chunks: list[Chunk] = []
    for pdf in pdfs:
        logger.info("ingesting %s", pdf.name)
        raw = extract_document(pdf)
        # Persist raw extraction BEFORE parsing (requirement 7).
        write_jsonl(settings.extracted_dir / f"{pdf.stem}.raw.jsonl", raw.pages)

        # Visual metadata (rendered pages + detection flags), kept separate.
        visual = extract_visuals(pdf, raw.meta.section, settings.visual_dir)
        write_jsonl(settings.visual_dir / f"{pdf.stem}.visual.jsonl", visual.pages)
        page_image_paths = {p.pdf_page_number: p.page_image_path for p in visual.pages}

        clean_pages = clean_document(raw)
        articles = parse_articles(raw.meta, clean_pages)
        # Persist parsed documents as JSONL before embedding (requirement 8).
        write_jsonl(settings.processed_dir / f"{pdf.stem}.articles.jsonl", articles)

        chunks = chunker_impl.chunk(articles, raw.meta, page_image_paths)
        write_jsonl(settings.processed_dir / f"{pdf.stem}.chunks.jsonl", chunks)
        all_chunks.extend(chunks)

    if not all_chunks:
        raise F1RagError("ingestion produced no chunks; check extraction/parsing")

    # Batch embedding (requirement) then build + persist the index.
    logger.info("embedding %d chunks with %s", len(all_chunks), embedder)
    vectors = embedder_impl.embed(
        [c.text for c in all_chunks], batch_size=settings.embedding_batch_size
    )
    store = _build_store(settings, index, idx_dir)
    store.build(all_chunks, vectors)
    idx_dir.mkdir(parents=True, exist_ok=True)
    store.save(idx_dir)
    # Always write a canonical chunks.jsonl (BM25/hybrid load chunks from here).
    write_jsonl(idx_dir / "chunks.jsonl", all_chunks)

    manifest = {
        "corpus_hash": corpus_hash,
        "chunker": chunker,
        "embedder": embedder,
        "embedding_model": settings.embedding_model if embedder == "minilm" else embedder,
        "embedding_dim": int(vectors.shape[1]) if vectors.ndim == 2 else 0,
        "index": index,
        "n_chunks": len(all_chunks),
        "n_documents": len(pdfs),
        "chunk_max_tokens": settings.chunk_max_tokens,
        "chunk_overlap_tokens": settings.chunk_overlap_tokens,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    logger.info("index built at %s (%d chunks)", idx_dir, len(all_chunks))
    return IngestResult(index_dir=idx_dir, n_chunks=len(all_chunks), n_documents=len(pdfs), skipped=False)


# ---------------------------------------------------------------------------
# Querying
# ---------------------------------------------------------------------------
@dataclass
class QueryEngine:
    settings: Settings
    chunks: list[Chunk]
    store: object
    embedder: object
    manifest: dict

    @classmethod
    def load(
        cls,
        settings: Settings,
        chunker: str | None = None,
        embedder: str | None = None,
        index: str | None = None,
    ) -> "QueryEngine":
        chunker = chunker or settings.chunker
        embedder_name = embedder or settings.embedder
        index_name = index or settings.index
        idx_dir = settings.indexes_dir / index_key(index_name, chunker, embedder_name)
        manifest_path = idx_dir / "manifest.json"
        if not manifest_path.exists():
            raise ConfigurationError(
                f"no index at {idx_dir}. Run `ingest` with matching flags first."
            )
        manifest = json.loads(manifest_path.read_text())
        chunks = read_jsonl(idx_dir / "chunks.jsonl", Chunk)
        embedder_impl = _build_embedder(settings, embedder_name)
        store = _build_store(settings, index_name, idx_dir)
        store.load(idx_dir)
        return cls(
            settings=settings,
            chunks=chunks,
            store=store,
            embedder=embedder_impl,
            manifest=manifest,
        )

    def _build_retriever(self, name: str):
        deps = RetrievalDeps(
            chunks=self.chunks,
            store=self.store,
            embedder=self.embedder,
            top_k=self.settings.top_k,
        )
        return retriever_registry.create(name, deps)

    def answer(
        self,
        question: str,
        retriever: str | None = None,
        reranker: str | None = None,
        section: str | None = None,
        k: int | None = None,
        generate: bool = True,
    ) -> tuple[Answer | None, QueryTrace]:
        """Run one query end-to-end and return (answer, trace).

        If ``generate`` is False (or no generator/API key), retrieval + context are
        still fully executed and traced, and the answer is returned as None.
        """

        retriever_name = retriever or self.settings.retriever
        reranker_name = reranker or self.settings.reranker
        k = k or self.settings.top_k

        rec = TraceRecorder()
        normalized = normalize_query(question)
        rec.record_query(question, normalized)
        rec.record_embedding(
            model=getattr(self.embedder, "name", None),
            dim=getattr(self.embedder, "dim", None),
            config={"batch_size": self.settings.embedding_batch_size},
        )

        metadata_filter: dict = {}
        if section:
            metadata_filter["section"] = RegulationSection(section.upper()).value
        rec.record_components(
            retriever=retriever_name,
            reranker=reranker_name,
            index=self.manifest.get("index", "?"),
            metadata_filters=metadata_filter,
            top_k=k,
        )

        # Retrieval -> reranking -> context assembly.
        retriever_impl = self._build_retriever(retriever_name)
        candidates = retriever_impl.retrieve(normalized, k=k, metadata_filter=metadata_filter or None)

        reranker_impl = reranker_registry.create(reranker_name)
        candidates = reranker_impl.rerank(normalized, candidates)
        rec.record_candidates(candidates)

        assembler = assembler_registry.create("greedy")
        context = assembler.assemble(candidates, max_tokens=self.settings.context_max_tokens)
        rec.record_context(context)

        answer: Answer | None = None
        if generate:
            try:
                generator = generator_registry.create(
                    self.settings.generator,
                    api_key=self.settings.anthropic_api_key,
                    model=self.settings.generation_model,
                    max_tokens=self.settings.generation_max_tokens,
                    temperature=self.settings.generation_temperature,
                )
                from .generation.prompts import get_prompt

                prompt = get_prompt()
                rec.record_prompt(prompt.version, prompt.render_user(normalized, context.text))
                answer = generator.generate(normalized, context)
                rec.record_answer(answer)
            except F1RagError as exc:
                logger.warning("generation skipped: %s", exc)

        return answer, rec.trace


def render_trace(trace: QueryTrace) -> str:
    return render_trace_text(trace)
