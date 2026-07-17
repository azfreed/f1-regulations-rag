# The RAG pipeline, stage by stage

This project breaks retrieval-augmented generation into explicit stages. Each is a
module with a typed input/output contract, a standalone implementation, and
diagnostics. You can run, test, and replace any stage without touching the others.

Data flows as inspectable artifacts: raw extraction and parsed chunks are written
to JSONL, indexes are persisted to disk, and every query produces a trace.

```
PDF ─► extract text ─► detect visuals ─► clean ─► parse ─► chunk ─► enrich
      ─► embed ─► index ─► retrieve ─► (keyword) ─► (rerank) ─► assemble context
      ─► generate ─► cite ─► evaluate ─► track experiment
```

## 1. PDF extraction (`extraction/text.py`)

Reads each page with PyMuPDF into a `RawPage` (raw text + positioned blocks) and
detects document metadata (issue number, publication date, WMSC date, section,
title) from the page-1 front matter and filename. Raw output is written to
`data/extracted/<section>.raw.jsonl` **before** any parsing, so you can always see
what came out of the PDF.

## 2. Visual asset detection + page rendering (`extraction/visuals.py`)

Renders every page to a PNG and flags pages likely to contain diagrams, vector
drawings, tables, equations, or embedded images (with bounding boxes when
available). Kept **separate** from text extraction. Multimodal retrieval is not
built yet, but the metadata to support it later is captured now. See
[visual-content.md](visual-content.md).

## 3. Document parsing / cleanup (`extraction/cleanup.py`)

Removes recurring headers/footers (detected by frequency + position, not brittle
per-file regexes) and flags table-of-contents pages so they are excluded from the
searchable corpus. Produces `CleanPage` objects.

## 4. Structural segmentation (`parsing/headings.py`, `parsing/regulations.py`)

Detects article/subarticle/provision boundaries and groups body text under the
right regulation reference, producing flat `Article` records
(`data/processed/<section>.articles.jsonl`). Splitting at article boundaries is
what lets chunks map to precise, citable references.

> Layout note: PyMuPDF emits the article *number* and its *heading/text* on
> separate lines, so the parser is a line stream with look-ahead rather than a
> per-line regex.

## 5. Chunking (`chunking/regulation.py`, `chunking/fixed_window.py`)

Turns articles into retrievable `Chunk`s with deterministic IDs.
- **regulation** (default): one chunk per article/subarticle/provision, splitting
  into overlapping windows only when a provision is too long.
- **fixed_window**: structure-agnostic fixed-size overlapping windows (baseline).

Comparing the two is a core experiment. Chunks are written to
`data/processed/<section>.chunks.jsonl`.

## 6. Metadata enrichment

Each chunk carries section, article number, parent article, heading, page number,
page label, source filename, document title, issue number, publication date, and
the page image path - everything needed for filtering and citation. (Performed
inline by the chunkers.)

## 7. Embedding (`embeddings/`)

Encodes chunk text into L2-normalized vectors (batched).
- **minilm** (default real model): `all-MiniLM-L6-v2` semantic embeddings.
- **hashing**: deterministic, offline, dependency-free bag-of-words vectors (great
  for tests and for seeing the mechanics without a model download).

Normalizing to unit length means cosine similarity becomes a dot product.

## 8. Indexing (`indexing/numpy_store.py`, `indexing/chroma_store.py`)

Stores vectors + chunk metadata and searches them.
- **numpy** (default): the transparent reference - cosine similarity is one matrix
  multiplication; read every line.
- **chroma**: a real vector DB with the same interface.

## 9. Candidate retrieval (`retrieval/vector.py`)

Embeds the query and returns the top-k most cosine-similar chunks, with metadata
filtering (e.g. restrict to one section).

## 10. Optional keyword retrieval (`retrieval/keyword.py`)

BM25 lexical retrieval - strong on exact terms, article numbers, and rare tokens
that embeddings blur.

## 11. Optional reranking (`reranking/identity.py`)

An explicit stage. Only a no-op (`none`) ships now; a cross-encoder reranker can be
dropped in later without changing retrieval or context.

Hybrid retrieval (`retrieval/hybrid.py`) fuses vector + BM25 by min-max normalizing
each score set and taking a weighted sum, deduplicating by chunk ID.

## 12. Context assembly (`context/assembly.py`)

Greedily selects the highest-ranked chunks under a token budget, deduplicates
near-duplicates, prefixes each with its citation label, and records what was
discarded and why.

## 13. Answer generation (`generation/anthropic_client.py`, `generation/prompts.py`)

Sends a versioned prompt + context to an Anthropic model at temperature 0. The
prompt forbids answering from general knowledge; if there is no context the
generator raises rather than guessing, and the model is instructed to reply
`UNANSWERABLE` when the context is insufficient.

## 14. Citation rendering (`generation/base.py`)

Parses the article labels the model actually cited and maps them back to the chunks
in context, so citations reflect what was used - the basis for citation-accuracy
metrics.

## 15. Evaluation (`evaluation/`)

A versioned dataset of questions with expected articles/terms and answerability.
Metrics: recall@k, MRR, article hit rate, citation accuracy, unsupported-claim
rate, unanswerable handling, and answer-term coverage.

## 16. Experiment tracking (`evaluation/runner.py`, `tracing/`)

Every evaluation run is saved to `experiments/runs/<timestamp>.json` with the full
configuration (corpus version, chunking config, embedding model, retriever,
reranker, generation model, prompt version) and metrics, so configurations are
directly comparable. Every query can also emit a full JSON trace under
`experiments/traces/`.

## A small worked example

For the question *"How is a Fillet Radius defined?"*, regulation-aware chunking
keeps Section C article **C3.2.6** intact:

> A "Fillet Radius" is formed by rounding an internal corner (included angle less
> than 180 degrees) with a concave surface by only adding material ...

so retrieval returns a single chunk that maps cleanly to the citation
`Art. C3.2.6 (Section C, p.C15)`. A fixed-window chunk of the same text might
straddle C3.2.5 and C3.2.6, producing a fuzzier citation - which is exactly the
kind of trade-off this test bed lets you measure.
