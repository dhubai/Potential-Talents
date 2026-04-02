"""Patch PotentialTalents.ipynb to make feedback rounds interactive."""
from __future__ import annotations

import json
from pathlib import Path


ROUND1_SOURCE = [
    "# --- Round 1: YOUR TURN --- star and skip candidates based on the ranking above.\n",
    "star_input = input('Round 1 - Enter candidate IDs to STAR (space-separated, e.g. 100 53): ').strip()\n",
    "skip_input = input('Round 1 - Enter candidate IDs to SKIP (space-separated, or blank): ').strip()\n",
    "\n",
    "starred_r1 = [int(x) for x in star_input.split() if x.isdigit()] if star_input else []\n",
    "skipped_r1 = [int(x) for x in skip_input.split() if x.isdigit()] if skip_input else []\n",
    "\n",
    "ranker.record_feedback(QUERIES, starred_ids=starred_r1, skipped_ids=skipped_r1)\n",
    "ranked_v1 = ranker.rank(QUERIES, top_k=50)\n",
    "IDEAL_IDS.update(starred_r1)\n",
    "m1 = evaluate_ranking(ranked_v1, IDEAL_IDS)\n",
    "action_str = f'star {starred_r1}, skip {skipped_r1}'\n",
    "history.append({'round': 1, 'action': action_str, **m1})\n",
    "rankings[1] = ranked_v1.copy()\n",
    "\n",
    "print(f'\\nRound 1 ({action_str}):')\n",
    "print(f'  NDCG@10 = {m1[\"ndcg_at_k\"]},  MAP@10 = {m1[\"map_at_k\"]}')\n",
    "for cid in sorted(IDEAL_IDS):\n",
    "    print(f'  Candidate {cid} at rank #{rank_position(ranked_v1, cid)}')\n",
    "print('\\nUpdated top 10:')\n",
    "ranked_v1[['id', 'job_title', 'fit', 'embed_similarity']].head(10)\n",
]

ROUND2_SOURCE = [
    "# --- Round 2: YOUR TURN --- another round of feedback.\n",
    "star_input2 = input('Round 2 - Enter candidate IDs to STAR (space-separated, or blank): ').strip()\n",
    "skip_input2 = input('Round 2 - Enter candidate IDs to SKIP (space-separated, or blank): ').strip()\n",
    "\n",
    "starred_r2 = [int(x) for x in star_input2.split() if x.isdigit()] if star_input2 else []\n",
    "skipped_r2 = [int(x) for x in skip_input2.split() if x.isdigit()] if skip_input2 else []\n",
    "\n",
    "ranker.record_feedback(QUERIES, starred_ids=starred_r2, skipped_ids=skipped_r2)\n",
    "ranked_v2 = ranker.rank(QUERIES, top_k=50)\n",
    "IDEAL_IDS.update(starred_r2)\n",
    "m2 = evaluate_ranking(ranked_v2, IDEAL_IDS)\n",
    "action_str2 = f'star {starred_r2}, skip {skipped_r2}'\n",
    "history.append({'round': 2, 'action': action_str2, **m2})\n",
    "rankings[2] = ranked_v2.copy()\n",
    "\n",
    "print(f'\\nRound 2 ({action_str2}):')\n",
    "print(f'  NDCG@10 = {m2[\"ndcg_at_k\"]},  MAP@10 = {m2[\"map_at_k\"]}')\n",
    "for cid in sorted(IDEAL_IDS):\n",
    "    print(f'  Candidate {cid} at rank #{rank_position(ranked_v2, cid)}')\n",
    "print('\\nUpdated top 10:')\n",
    "ranked_v2[['id', 'job_title', 'fit', 'embed_similarity']].head(10)\n",
]


def main() -> None:
    p = Path("PotentialTalents.ipynb")
    nb = json.loads(p.read_text(encoding="utf-8"))

    patched = 0
    for cell in nb["cells"]:
        src = "".join(cell.get("source", []))

        if "# --- Round 1: Star candidate 100" in src:
            cell["source"] = ROUND1_SOURCE
            cell["outputs"] = []
            cell["execution_count"] = None
            patched += 1

        elif "# --- Round 2: Star candidate 53" in src:
            cell["source"] = ROUND2_SOURCE
            cell["outputs"] = []
            cell["execution_count"] = None
            patched += 1

    p.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Patched {patched} cells.")


if __name__ == "__main__":
    main()
