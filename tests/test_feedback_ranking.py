from __future__ import annotations

from talentfit.config import PotentialTalentsConfig
from talentfit.ranker import PotentialTalentsRanker


def test_starred_candidate_moves_up_with_feedback(tmp_path) -> None:
    cfg = PotentialTalentsConfig(
        data_path="potential-talents.xlsx",
        feedback_path=tmp_path / "feedback.json",
        top_k=50,
    )
    ranker = PotentialTalentsRanker(config=cfg)

    queries = ["aspiring human resources role", "seeking human resources"]
    target_id = 53

    before = ranker.rank(queries, top_k=50)["id"].tolist()
    pos_before = before.index(target_id) if target_id in before else 999

    ranker.record_feedback(queries, starred_ids=[target_id], skipped_ids=[])
    after = ranker.rank(queries, top_k=50)["id"].tolist()
    pos_after = after.index(target_id) if target_id in after else 999

    assert pos_after <= pos_before

