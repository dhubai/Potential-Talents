from __future__ import annotations

import argparse

import numpy as np

from talentfit.cutoff import CutoffPolicy
from talentfit.reporting import simple_impact_report
from talentfit.ranker import PotentialTalentsRanker


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--query", action="append", required=True)
    p.add_argument("--topk", type=int, default=20)
    args = p.parse_args()

    ranker = PotentialTalentsRanker()
    ranked = ranker.rank(args.query, top_k=max(args.topk, 50))
    cutoff = CutoffPolicy().decide(ranked["fit"].to_numpy())
    ranked["keep"] = (ranked["fit"] >= cutoff).astype(int)

    cols = [
        "id",
        "job_title",
        "fit",
        "keep",
        "embed_similarity",
        "is_out_of_scope",
        "connections",
        "location",
    ]
    print("cutoff_fit:", float(np.round(cutoff, 4)))
    rep = simple_impact_report(ranked)
    print(f"kept_rate: {rep.kept_rate:.2%} ({rep.n_kept}/{rep.n_total})")
    print(ranked[cols].head(args.topk).to_string(index=False))


if __name__ == "__main__":
    main()

