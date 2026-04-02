from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PotentialTalentsConfig:
    data_path: Path = Path("potential-talents.xlsx")
    cache_dir: Path = Path(".cache") / "talentfit"
    feedback_path: Path = Path(".cache") / "talentfit" / "feedback.json"

    # Retrieval defaults
    top_k: int = 50

    # Legacy baseline (parity) defaults
    legacy_bert_model_name: str = "bert-base-uncased"
    legacy_doc2vec_vector_size: int = 768
    legacy_doc2vec_window: int = 2
    legacy_doc2vec_epochs: int = 40

    # Modern embeddings default (used in later todos)
    embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
