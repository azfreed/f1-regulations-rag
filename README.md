# f1-regulations-rag

A local, inspectable **RAG test bed** over the 2026 FIA Formula 1 Regulations.

The goal is to create a sandbox for Retrieval Augmented Generation where **every stage is
visible, independently testable, and swappable** so you can learn about and
experiment with retrieval-augmented generation on a real, structured corpus. 
In this case, the chosen corpus is the 2026 F1 Regulations.

Included:

- 16 explicit pipeline stages, each with a typed interface and a standalone
  implementation (see [docs/pipeline.md](docs/pipeline.md)).
- Alternative implementations selectable at runtime via registries: regulation-aware
  vs fixed-window chunking, MiniLM vs a hashing embedder, a transparent **NumPy**
  vector index vs **Chroma**, **vector / BM25 / hybrid** retrieval, optional
  reranking, and versioned prompts.
- A full **query trace** for every question (raw distances, similarities, keyword
  scores, selected/discarded chunks, the exact prompt, and citations).
- A versioned **evaluation** harness (recall@k, MRR, article hit rate, citation
  accuracy, unsupported-claim rate, unanswerable handling) with comparable
  experiment records.

## Requirements

- Python 3.11+
- The six regulation PDFs in `data/raw/` (already present):
  `section-a-general.pdf`, `section-b-sporting.pdf`, `section-c-technical.pdf`,
  `section-d-financial-teams.pdf`, `section-e-financial-pu.pdf`,
  `section-f-operational.pdf`.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate

# Core (offline) install: extraction, chunking, NumPy index, BM25, tracing, eval.
pip install -e ".[dev]"

# Full install: semantic embeddings (MiniLM), Chroma, and Anthropic generation.
pip install -e ".[dev,embeddings,chroma,anthropic]"

cp .env.example .env   # optional; set ANTHROPIC_API_KEY to enable generation
```

The system runs end-to-end **offline** using the `hashing` embedder + `numpy`
index. Semantic retrieval (`minilm`) downloads a small model on first use;
answer generation requires `ANTHROPIC_API_KEY`.

## Usage

Inspect the corpus before indexing (Milestone 1 deliverable):

```bash
python -m f1_rag.cli diagnose            # writes experiments/diagnostics/*.md + *.json
```

List the implementations available for each stage:

```bash
python -m f1_rag.cli components
```

Build an index (idempotent; re-running with an unchanged corpus is a no-op):

```bash
python -m f1_rag.cli ingest \
  --chunker regulation \
  --embedder minilm \
  --index chroma
```

Ask a question (prints a full trace; add `--json-trace` to save it):

```bash
python -m f1_rag.cli ask \
  --retriever hybrid \
  --reranker none \
  "What restrictions apply during a factory shutdown?"
```

Retrieval-only (no API key needed), restricted to one section:

```bash
python -m f1_rag.cli ask --embedder minilm --retriever hybrid \
  --no-generate --section D "What is included in the cost cap?"
```

Run the evaluation dataset and save a comparable experiment record:

```bash
python -m f1_rag.cli evaluate --retriever hybrid            # retrieval metrics
python -m f1_rag.cli evaluate --retriever hybrid --generate # + answer metrics
```

> The `ask` / `evaluate` commands must use the **same** `--chunker/--embedder/--index`
> flags used at `ingest` time, because each configuration is stored in its own
> index directory (`indexes/<index>__<chunker>__<embedder>/`).

## Project layout

```
data/raw/         source PDFs (input)
data/extracted/   raw per-page extraction (JSONL, pre-parsing)
data/processed/   parsed articles + chunks (JSONL, pre-embedding)
data/visual/      rendered page images + visual metadata
data/evaluations/ versioned evaluation datasets
indexes/          persisted vector indexes (per configuration)
experiments/      diagnostics, query traces, and evaluation runs
src/f1_rag/       the pipeline (one package per stage)
tests/            unit tests per stage
docs/             architecture, pipeline, experiments, visual-content
```

## Documentation

- [docs/pipeline.md](docs/pipeline.md) - every RAG stage in plain language.
- [docs/architecture.md](docs/architecture.md) - module boundaries and data flow.
- [docs/experiments.md](docs/experiments.md) - comparing configurations.
- [docs/visual-content.md](docs/visual-content.md) - current + planned image handling.

## Status

Milestone 1 is complete: project structure, typed models, text extraction, page
rendering + visual metadata, heading detection, per-PDF diagnostics, and the docs.
The full pipeline (chunking through generation, plus a NumPy retriever) is also
implemented so the mechanics can be explored end-to-end.
