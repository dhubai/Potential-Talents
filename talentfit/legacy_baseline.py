from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModel, AutoTokenizer

from .config import PotentialTalentsConfig
from .text import TextPreprocessor


@dataclass
class LegacyBaselineRanker:
    """
    Legacy baseline that mirrors the original notebook:
    - Clean `job_title` into `job_title_cleaned`
    - Compute mean-pooled BERT embeddings (bert-base-uncased)
    - Train Doc2Vec on cleaned titles and infer vectors
    - Rank by cosine similarity to (possibly multi-)query keywords

    Notes:
    - Uses PyTorch Transformers instead of TF; scores won't be bit-identical,
      but rankings should be very similar.
    """

    config: PotentialTalentsConfig = PotentialTalentsConfig()
    device: str | None = None
    seed: int = 42

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        self._device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._text = TextPreprocessor()
        self._tokenizer = AutoTokenizer.from_pretrained(self.config.legacy_bert_model_name)
        self._bert = AutoModel.from_pretrained(self.config.legacy_bert_model_name).to(self._device)
        self._bert.eval()

    def load_data(self, path: str | None = None) -> pd.DataFrame:
        data_path = path or str(self.config.data_path)
        if data_path.lower().endswith(".xlsx"):
            df = pd.read_excel(data_path)
        else:
            df = pd.read_csv(data_path)
        return df

    def add_clean_title(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["job_title_cleaned"] = out["job_title"].astype(str).map(self._text.clean_sentence)
        return out

    @torch.inference_mode()
    def _bert_mean_pool(self, texts: Sequence[str]) -> np.ndarray:
        # Batch tokenize for speed.
        enc = self._tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        enc = {k: v.to(self._device) for k, v in enc.items()}
        out = self._bert(**enc)
        token_embeddings = out.last_hidden_state  # (B, T, H)
        attention_mask = enc["attention_mask"].unsqueeze(-1)  # (B, T, 1)
        masked = token_embeddings * attention_mask
        summed = masked.sum(dim=1)  # (B, H)
        counts = attention_mask.sum(dim=1).clamp(min=1)  # (B, 1)
        mean = summed / counts
        return mean.detach().cpu().numpy()

    def _train_doc2vec(self, cleaned_titles: Sequence[str]) -> Doc2Vec:
        tagged = [
            TaggedDocument(words=title.split(), tags=[str(i)])
            for i, title in enumerate(cleaned_titles)
        ]
        model = Doc2Vec(
            vector_size=self.config.legacy_doc2vec_vector_size,
            window=self.config.legacy_doc2vec_window,
            min_count=1,
            workers=1,  # deterministic
            epochs=self.config.legacy_doc2vec_epochs,
            seed=self.seed,
        )
        model.build_vocab(tagged)
        model.train(tagged, total_examples=model.corpus_count, epochs=model.epochs)
        return model

    def _doc2vec_embed(self, model: Doc2Vec, cleaned_titles: Sequence[str]) -> np.ndarray:
        vectors: list[np.ndarray] = []
        for title in cleaned_titles:
            vectors.append(model.infer_vector(title.split(), epochs=20, alpha=0.025))
        return np.asarray(vectors)

    def _prepare_queries(self, queries: Iterable[str]) -> list[str]:
        cleaned: list[str] = []
        for q in queries:
            q2 = self._text.replace_abbreviations(str(q))
            cleaned.append(self._text.clean_sentence(q2))
        return cleaned

    def rank(
        self,
        df: pd.DataFrame,
        queries: Sequence[str],
    ) -> pd.DataFrame:
        data = self.add_clean_title(df)
        query_clean = self._prepare_queries(queries)

        # Embeddings
        q_bert = self._bert_mean_pool(query_clean).mean(axis=0, keepdims=True)
        c_bert = self._bert_mean_pool(data["job_title_cleaned"].tolist())

        d2v = self._train_doc2vec(data["job_title_cleaned"].tolist())
        q_d2v = self._doc2vec_embed(d2v, query_clean).mean(axis=0, keepdims=True)
        c_d2v = self._doc2vec_embed(d2v, data["job_title_cleaned"].tolist())

        # Similarities
        bert_sim = cosine_similarity(q_bert, c_bert)[0]
        d2v_sim = cosine_similarity(q_d2v, c_d2v)[0]

        out = data.copy()
        out["bert_similarity"] = bert_sim
        out["doc2vec_similarity"] = d2v_sim
        out["mean_score"] = out[["bert_similarity", "doc2vec_similarity"]].mean(axis=1)
        out = out.sort_values("mean_score", ascending=False).reset_index(drop=True)
        return out

    def rerank_with_starred(
        self,
        ranked_df: pd.DataFrame,
        starred_ids: Sequence[int],
    ) -> pd.DataFrame:
        data = ranked_df.copy()
        data["is_starred"] = data["id"].isin(list(starred_ids)).astype(int)
        starred_queries = data.loc[data["is_starred"] == 1, "job_title_cleaned"].tolist()

        if not starred_queries:
            data["starred_similarity"] = 0.0
            data["mean_similarity_bert"] = data["bert_similarity"]
            data["mean_similarity_doc2vec"] = data["doc2vec_similarity"]
            return data

        # Compute starred similarity as mean similarity to starred titles using the same rank() path
        # (This mirrors the notebook's repeated encode_and_get_similarity calls conceptually).
        # For simplicity, we recompute a BERT embedding for all titles and the starred titles.
        c_bert = self._bert_mean_pool(data["job_title_cleaned"].tolist())
        s_bert = self._bert_mean_pool(starred_queries)
        s_mean = s_bert.mean(axis=0, keepdims=True)
        starred_sim = cosine_similarity(s_mean, c_bert)[0]

        data["starred_similarity"] = starred_sim
        data["mean_similarity_bert"] = data[["bert_similarity", "starred_similarity"]].mean(axis=1)
        data["mean_similarity_doc2vec"] = data[["doc2vec_similarity", "starred_similarity"]].mean(axis=1)

        # Prefer starred items on ties (like notebook sorting by mean + is_starred)
        data = data.sort_values(
            by=["mean_similarity_bert", "is_starred"],
            ascending=[False, False],
        ).reset_index(drop=True)
        return data

