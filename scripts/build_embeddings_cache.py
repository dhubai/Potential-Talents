from __future__ import annotations

from talentfit.config import PotentialTalentsConfig
from talentfit.embed import Embedder
from talentfit.ingest import load_candidates
from talentfit.text import TextPreprocessor


def main() -> None:
    cfg = PotentialTalentsConfig()
    res = load_candidates(cfg.data_path)
    df = res.df

    text = TextPreprocessor()
    cleaned = df["job_title"].astype(str).map(text.clean_sentence).tolist()

    emb = Embedder(cfg).embed(cleaned)
    print("cached embeddings:", emb.shape)


if __name__ == "__main__":
    main()

