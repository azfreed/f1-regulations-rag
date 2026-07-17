# Experiments: comparing configurations

This test bed is built to answer questions like *"does hybrid retrieval beat pure
vector search on these regulations?"* or *"does regulation-aware chunking improve
citation accuracy over fixed windows?"* - with numbers, not vibes.

## The knobs

| Stage | Flag | Options |
| --- | --- | --- |
| chunking | `--chunker` | `regulation`, `fixed_window` |
| embedding | `--embedder` | `minilm`, `hashing` |
| index | `--index` | `numpy`, `chroma` |
| retrieval | `--retriever` | `vector`, `bm25`, `hybrid` |
| reranking | `--reranker` | `none` |
| generation | prompt version | `v1` (see `generation/prompts.py`) |

Each `(index, chunker, embedder)` combination is a separate index directory, so
they never clobber each other. Retrieval/rerank/prompt choices are made at query
time against an existing index.

## Metrics (`evaluation/metrics.py`)

- **recall@k** - fraction of a case's expected articles present in the top-k
  candidates (prefix match: expecting `D3` is satisfied by `D3.1`).
- **MRR** - mean reciprocal rank of the first correct article.
- **article hit rate** - fraction of cases with at least one expected article in
  the top-k.
- **citation accuracy** - of cases that produced citations, fraction whose
  citations match the expected articles.
- **unsupported-claim rate** - fraction of substantive answers that either cite
  nothing or cite an article not present in the assembled context.
- **unanswerable handling** - fraction of cases handled correctly (unanswerable
  questions declined, answerable questions not wrongly declined).
- **answer-term coverage** - light lexical check that expected key terms appear.

## Running a comparison

Build the two chunkers you want to compare (embed once each), then evaluate:

```bash
python -m f1_rag.cli ingest --chunker regulation   --embedder minilm --index numpy
python -m f1_rag.cli ingest --chunker fixed_window --embedder minilm --index numpy

python -m f1_rag.cli evaluate --chunker regulation   --embedder minilm --index numpy --retriever hybrid
python -m f1_rag.cli evaluate --chunker fixed_window --embedder minilm --index numpy --retriever hybrid
```

Compare retrievers against one index:

```bash
python -m f1_rag.cli evaluate --retriever vector
python -m f1_rag.cli evaluate --retriever bm25
python -m f1_rag.cli evaluate --retriever hybrid
```

Add `--generate` to also measure citation accuracy and unsupported-claim rate
(requires `ANTHROPIC_API_KEY`).

## Reading the results

Every run writes `experiments/runs/<timestamp>.json`:

```json
{
  "timestamp": "...",
  "dataset_version": "2026-seed-v1",
  "corpus_version": "<hash>",
  "config": {
    "chunker": "regulation", "embedder": "minilm", "index": "numpy",
    "retriever": "hybrid", "reranker": "none",
    "generation_model": null, "prompt_version": null, "k": 8
  },
  "metrics": { "recall_at_k": 0.83, "mrr": 0.71, ... },
  "per_case": [ ... ]
}
```

Because every record has the same shape, you can load a directory of runs into a
DataFrame and pivot on `config.*` to compare metrics across configurations. The
`per_case` array lets you drill into which questions a configuration got wrong.

## The evaluation dataset

The seed dataset (`evaluation/dataset.py`, version `2026-seed-v1`) contains
grounded questions with expected articles (e.g. `C3.2.6` for the Fillet Radius
definition) and two intentionally **unanswerable** questions (ticket prices, a
driver salary cap) to exercise the decline behavior. Extend it by writing a new
versioned JSON file under `data/evaluations/` and passing `--dataset`.
