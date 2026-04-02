from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from .config import PotentialTalentsConfig
from .embed import Embedder
from .feedback import FeedbackStore
from .features import FeatureWeights, build_features, score_fit
from .ingest import load_candidates
from .retrieve import cosine_topk
from .text import TextPreprocessor


@dataclass
class PotentialTalentsRanker:
    config: PotentialTalentsConfig = PotentialTalentsConfig()
    weights: FeatureWeights = FeatureWeights()

    def __post_init__(self) -> None:
        self._text = TextPreprocessor()
        self._embed = Embedder(self.config)
        self._feedback = FeedbackStore(self.config.feedback_path)

    def _feedback_profile(self, queries: Sequence[str]) -> tuple[list[int], list[int]]:
        """
        Return (starred_ids, skipped_ids) for a query, using simple exact match on joined query string.
        """
        data = self._feedback.load()
        key = " | ".join([q.strip().lower() for q in queries])
        starred: list[int] = []
        skipped: list[int] = []
        for ev in data.get("events", []):
            if ev.get("query_key") != key:
                continue
            starred.extend([int(x) for x in ev.get("starred_ids", [])])
            skipped.extend([int(x) for x in ev.get("skipped_ids", [])])
        # de-dupe preserving order
        def _uniq(xs: list[int]) -> list[int]:
            seen = set()
            out: list[int] = []
            for x in xs:
                if x in seen:
                    continue
                seen.add(x)
                out.append(x)
            return out
        return _uniq(starred), _uniq(skipped)

    def rank(self, queries: Sequence[str], *, top_k: int | None = None) -> pd.DataFrame:
        res = load_candidates(self.config.data_path)
        df = res.df.copy()

        cleaned_titles = df["job_title"].astype(str).map(self._text.clean_sentence).tolist()
        doc_emb = self._embed.embed(cleaned_titles, normalize=True)  # (n, d)

        q_clean = [self._text.clean_sentence(self._text.replace_abbreviations(q)) for q in queries]
        q_emb = self._embed.embed(q_clean, normalize=True).mean(axis=0)  # (d,)

        # Rocchio-style relevance feedback: adjust query embedding toward starred and away from skipped.
        starred_ids, skipped_ids = self._feedback_profile(list(queries))
        if starred_ids or skipped_ids:
            id_to_row = {int(i): idx for idx, i in enumerate(df["id"].tolist())}
            alpha, beta, gamma = 1.0, 0.75, 0.25
            adj = alpha * q_emb
            if starred_ids:
                rows = [id_to_row[i] for i in starred_ids if i in id_to_row]
                if rows:
                    adj = adj + beta * doc_emb[rows].mean(axis=0)
            if skipped_ids:
                rows = [id_to_row[i] for i in skipped_ids if i in id_to_row]
                if rows:
                    adj = adj - gamma * doc_emb[rows].mean(axis=0)
            # renormalize
            norm = float(np.linalg.norm(adj) + 1e-12)
            q_emb = (adj / norm).astype(np.float32)

        k = top_k or self.config.top_k
        idx, sim = cosine_topk(q_emb, doc_emb, k=k)

        subset = df.iloc[idx].copy().reset_index(drop=True)
        subset["embed_similarity"] = sim.astype(np.float32)

        feats = build_features(
            embed_sim=subset["embed_similarity"].to_numpy(),
            query_text=" ".join(queries),
            doc_texts=subset["job_title"].astype(str).tolist(),
            connections=subset["connections"].astype("float").to_numpy(),
        )
        subset["fit"] = score_fit(feats, self.weights)

        # Basic out-of-scope filter signal: low semantic + low lexical overlap.
        subset["out_of_scope_score"] = (1.0 - subset["embed_similarity"]).clip(0.0, 1.0)
        subset["lexical_hint"] = feats["lex_sim"]
        subset["is_out_of_scope"] = (
            (subset["embed_similarity"] < 0.35) & (subset["lexical_hint"] < 0.05)
        ).astype(int)

        subset = subset.sort_values("fit", ascending=False).reset_index(drop=True)
        return subset

    def record_feedback(
        self,
        queries: Sequence[str],
        *,
        starred_ids: Sequence[int] = (),
        skipped_ids: Sequence[int] = (),
    ) -> None:
        key = " | ".join([q.strip().lower() for q in queries])
        self._feedback.append_event(
            {
                "query_key": key,
                "starred_ids": list(map(int, starred_ids)),
                "skipped_ids": list(map(int, skipped_ids)),
            }
        )

