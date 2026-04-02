from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

from .config import PotentialTalentsConfig


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class Embedder:
    config: PotentialTalentsConfig = PotentialTalentsConfig()
    device: str | None = None

    def __post_init__(self) -> None:
        self._model = SentenceTransformer(
            self.config.embed_model_name,
            device=self.device,
        )
        self._cache_dir = self.config.cache_dir / "embeddings"
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_key(self, texts: Sequence[str]) -> str:
        payload = {
            "model": self.config.embed_model_name,
            "texts_hash": _sha256_text("\n".join(texts)),
        }
        return _sha256_text(json.dumps(payload, sort_keys=True))

    def embed(self, texts: Sequence[str], *, normalize: bool = True) -> np.ndarray:
        key = self._cache_key(texts)
        path = self._cache_dir / f"{key}.npz"
        if path.exists():
            return np.load(path)["embeddings"]

        emb = self._model.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=normalize,
            show_progress_bar=False,
        ).astype(np.float32)
        np.savez_compressed(path, embeddings=emb)
        return emb

