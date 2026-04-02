from __future__ import annotations

import argparse

import numpy as np

from talentfit.eval import map_at_k, ndcg_at_k
from talentfit.ranker import PotentialTalentsRanker


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--query", action="append", required=True)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--ideal", type=int, nargs="*", default=[])
    p.add_argument("--star", type=int, nargs="*", default=[])
    p.add_argument("--skip", type=int, nargs="*", default=[])
    args = p.parse_args()

    ranker = PotentialTalentsRanker()

    # Baseline ranking
    ranked0 = ranker.rank(args.query, top_k=50)
    ids0 = ranked0["id"].tolist()
    rel0 = np.array([1 if i in set(args.ideal) else 0 for i in ids0], dtype=int)
    print("Before feedback:", "NDCG@k", round(ndcg_at_k(rel0, args.k), 4), "MAP@k", round(map_at_k(rel0, args.k), 4))

    # Apply feedback then rerank
    if args.star or args.skip:
        ranker.record_feedback(args.query, starred_ids=args.star, skipped_ids=args.skip)

    ranked1 = ranker.rank(args.query, top_k=50)
    ids1 = ranked1["id"].tolist()
    rel1 = np.array([1 if i in set(args.ideal) else 0 for i in ids1], dtype=int)
    print("After feedback: ", "NDCG@k", round(ndcg_at_k(rel1, args.k), 4), "MAP@k", round(map_at_k(rel1, args.k), 4))

    print("\nTop-10 before:", ids0[:10])
    print("Top-10 after: ", ids1[:10])


if __name__ == "__main__":
    main()

