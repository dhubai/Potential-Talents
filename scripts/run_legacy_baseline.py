from __future__ import annotations

import argparse

from talentfit.legacy_baseline import LegacyBaselineRanker


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="potential-talents.xlsx")
    p.add_argument(
        "--query",
        action="append",
        default=["aspiring human resources", "seeking human resources"],
        help="Repeatable. Example: --query 'full stack engineer'",
    )
    p.add_argument("--topk", type=int, default=20)
    p.add_argument("--star", type=int, nargs="*", default=[])
    args = p.parse_args()

    ranker = LegacyBaselineRanker()
    df = ranker.load_data(args.data)
    ranked = ranker.rank(df, args.query)

    print("\nTop candidates by mean_score:")
    cols = ["id", "job_title", "mean_score", "bert_similarity", "doc2vec_similarity"]
    print(ranked[cols].head(args.topk).to_string(index=False))

    if args.star:
        reranked = ranker.rerank_with_starred(ranked, args.star)
        print("\nRe-ranked with starred ids:", args.star)
        cols = ["id", "job_title", "is_starred", "mean_similarity_bert", "starred_similarity"]
        print(reranked[cols].head(args.topk).to_string(index=False))


if __name__ == "__main__":
    main()

