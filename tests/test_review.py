"""Last-week review: snapshots, scoring with autosubs, and the written comments."""

from fpl_engine.recommend.review import (
    _comments,
    _score,
    _squad_from_picks,
    load_snapshot,
    write_snapshot,
)
from fpl_engine.rules.constants import rules_for


def test_snapshot_round_trip(tmp_path):
    write_snapshot(7, {1: 4.123456, 2: 0.5}, [{"id": 1, "role": "XI", "order": 1}], "t", root=tmp_path)
    snap = load_snapshot(7, root=tmp_path)
    assert snap["gw"] == 7 and snap["xp"] == {"1": 4.123, "2": 0.5}
    assert snap["recommended"][0]["id"] == 1
    assert load_snapshot(8, root=tmp_path) is None


def _picks():
    # 15 players: ids 1-2 GK, 3-7 DEF, 8-12 MID, 13-15 FWD; pick order = id
    pos = {i: 1 for i in (1, 2)} | {i: 2 for i in range(3, 8)} | {i: 3 for i in range(8, 13)} | {i: 4 for i in (13, 14, 15)}
    # XI: GK 1, DEF 3-5, MID 8-11, FWD 13-15 (3-4-3); bench: GK 2, then 6, 7, 12
    order = [1, 3, 4, 5, 8, 9, 10, 11, 13, 14, 15, 2, 6, 7, 12]
    picks = [
        {"element": e, "position": n, "is_captain": e == 13, "is_vice_captain": e == 14}
        for n, e in enumerate(order, start=1)
    ]
    return picks, pos


def test_score_applies_autosub_and_captain_double():
    rules = rules_for("2026-27")
    picks, pos = _picks()
    squad = _squad_from_picks(picks, pos, {e: 1 for e in pos})
    pts = {e: 2 for e in pos} | {13: 5, 6: 7}
    mins = {e: 90 for e in pos} | {4: 0}  # a starting defender does not play
    total, cap = _score(squad, pts, mins, rules)
    # After the autosub (defender 4 out, bench defender 6 in): GK 2 + DEF 2+2+7 + MID 4x2 + FWD 5+2+2 = 30,
    # plus the captain's 5 again.
    assert cap == 13
    assert total == 35


def test_comments_are_honest_when_the_old_squad_scored_more():
    review = {
        "event": 6, "points": 60, "gross_points": 60, "hit": 0, "bench_points": 0,
        "average": 55, "rank_before": 100, "rank_after": 90, "chip": "wildcard",
        "xi_xp": 55.7, "provisional": False,
        "players": [
            {"id": 1, "name": "A", "role": "captain", "points": 1, "xp": 5.9, "minutes": 90},
            {"id": 2, "name": "B", "role": "XI", "points": 15, "xp": 4.3, "minutes": 90},
        ],
        "what_if": [{"label": "Your GW5 squad, no wildcard", "points": 64}],
        "recommended_overlap": None,
        "model": {"n": 300, "mae": 1.9, "top10_actual": 5.5, "league_actual": 2.8},
    }
    text = " ".join(_comments(review))
    assert "5 above the average of 55" in text
    assert "1 point (doubled)" in text  # singular
    assert "did not pay off straight away" in text
    assert "B (15) would have paid far more" in text
