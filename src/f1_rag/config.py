"""Configuration and path resolution.

Why this exists
---------------
No stage should hard-code absolute paths or component choices. ``Settings`` reads
environment variables (and an optional ``.env`` file) with sensible defaults, and
resolves all data directories relative to a configurable project root. The CLI
passes a ``Settings`` instance down to stages, so tests can construct one pointed
at a temp directory.

Assumptions
-----------
- All paths are relative to ``project_root`` unless given as absolute paths.
- Temperature for generation is fixed to 0 (see ``generation_temperature``); it is
  not configurable, by design, for reproducibility.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _default_project_root() -> Path:
    # The package lives at <root>/src/f1_rag/config.py -> root is parents[2].
    return Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Runtime settings, populated from env vars prefixed ``F1RAG_`` and ``.env``."""

    model_config = SettingsConfigDict(
        env_prefix="F1RAG_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Paths ---
    project_root: Path = Field(default_factory=_default_project_root)
    data_dir: Path = Path("data")
    index_dir: Path = Path("indexes")
    experiment_dir: Path = Path("experiments")

    # --- Default component names (CLI flags override these) ---
    chunker: str = "regulation"
    embedder: str = "minilm"
    index: str = "numpy"
    retriever: str = "vector"
    reranker: str = "none"
    generator: str = "anthropic"

    # --- Embeddings ---
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_batch_size: int = 32

    # --- Chunking ---
    chunk_max_tokens: int = 350
    chunk_overlap_tokens: int = 60

    # --- Retrieval / context ---
    top_k: int = 8
    context_max_tokens: int = 3000

    # --- Generation ---
    # ANTHROPIC_API_KEY is read without the F1RAG_ prefix (Anthropic convention).
    anthropic_api_key: str | None = Field(default=None, alias="ANTHROPIC_API_KEY")
    generation_model: str = "claude-3-5-sonnet-20241022"
    generation_max_tokens: int = 1024
    generation_temperature: float = 0.0  # fixed by policy; do not expose to CLI

    # ------------------------------------------------------------------
    # Resolved paths. Each is created lazily via ``ensure_dirs`` when writing.
    # ------------------------------------------------------------------
    def _resolve(self, p: Path) -> Path:
        return p if p.is_absolute() else (self.project_root / p)

    @property
    def raw_dir(self) -> Path:
        return self._resolve(self.data_dir) / "raw"

    @property
    def extracted_dir(self) -> Path:
        return self._resolve(self.data_dir) / "extracted"

    @property
    def processed_dir(self) -> Path:
        return self._resolve(self.data_dir) / "processed"

    @property
    def visual_dir(self) -> Path:
        return self._resolve(self.data_dir) / "visual"

    @property
    def evaluations_dir(self) -> Path:
        return self._resolve(self.data_dir) / "evaluations"

    @property
    def indexes_dir(self) -> Path:
        return self._resolve(self.index_dir)

    @property
    def experiments_dir(self) -> Path:
        return self._resolve(self.experiment_dir)

    @property
    def diagnostics_dir(self) -> Path:
        return self.experiments_dir / "diagnostics"

    @property
    def traces_dir(self) -> Path:
        return self.experiments_dir / "traces"

    @property
    def runs_dir(self) -> Path:
        return self.experiments_dir / "runs"

    def ensure_dirs(self) -> None:
        """Create all output directories if missing (idempotent)."""

        for d in (
            self.extracted_dir,
            self.processed_dir,
            self.visual_dir,
            self.evaluations_dir,
            self.indexes_dir,
            self.diagnostics_dir,
            self.traces_dir,
            self.runs_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)
