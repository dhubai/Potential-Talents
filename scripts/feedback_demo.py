from __future__ import annotations

import argparse

from talentfit.ranker import PotentialTalentsRanker


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--query", action="append", required=True)
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--star", type=int, nargs="*", default=[])
    p.add_argument("--skip", type=int, nargs="*", default=[])
    args = p.parse_args()

    ranker = PotentialTalentsRanker()

    if args.star or args.skip:
        ranker.record_feedback(args.query, starred_ids=args.star, skipped_ids=args.skip)
        print("Recorded feedback.")

    ranked = ranker.rank(args.query, top_k=args.topk)
    cols = ["id", "job_title", "fit", "embed_similarity", "connections"]
    print(ranked[cols].to_string(index=False))


if __name__ == "__main__":
    main()

