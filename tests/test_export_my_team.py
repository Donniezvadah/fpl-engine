"""The My Team section of the dashboard payload: roles, wildcard moves, budget."""

import pandas as pd

from fpl_engine.optim.chips import ChipEvaluation
from fpl_engine.optim.planner import PlannedGameweek, TransferPlan
from fpl_engine.recommend.export import _my_team_payload
from fpl_engine.recommend.weekly import MyTeamInfo, WeeklyReport


def _player(pid: int, pos: int, price: int, xp: float) -> dict:
    return {
        "id": pid,
        "name": f"P{pid}",
        "position": pos,
        "club_short": "AAA",
        "price": price,
        "xp": xp,
        "xp_by_gw": {"6": xp, "7": xp},
    }


def _report(my_team: MyTeamInfo | None) -> WeeklyReport:
    return WeeklyReport(
        season="2026-27",
        next_gw=6,
        mode="linked",
        xp_table=pd.DataFrame(),
        plan=None,
        draft=None,
        captains=[],
        chips=[],
        differentials=pd.DataFrame(),
        price_watch=pd.DataFrame(),
        names={i: f"P{i}" for i in range(1, 40)},
        horizon_gws=[6, 7],
        my_team=my_team,
    )


def test_draft_mode_has_no_my_team():
    assert _my_team_payload(_report(None), []) is None


def test_wildcard_moves_pair_by_position_and_roles_are_assigned():
    players = [
        _player(1, 1, 50, 3.0), _player(2, 2, 50, 2.0), _player(3, 2, 45, 4.0),
        _player(4, 3, 80, 5.0), _player(5, 4, 90, 6.0),
    ]
    picks = pd.DataFrame(
        {
            "player_id": [1, 2, 4, 5],
            "pick_order": [1, 2, 3, 4],
            "is_captain": [False, False, False, True],
            "is_vice_captain": [False, False, True, False],
        }
    )
    plan = TransferPlan(
        gameweeks=[
            PlannedGameweek(
                event=6, transfers_in=[3], transfers_out=[2], hits=0,
                free_transfers_before=1, starting_ids=[1, 3, 4, 5], captain_id=5,
                bench_ids=[], xp=0.0,
            )
        ],
        objective=10.0,
    )
    mt = MyTeamInfo(
        squad={1: 50, 2: 48, 4: 80, 5: 90},
        bank=2, free_transfers=1, team_value=270,
        chips_used=[("3xc", 3)], chips_available=["wildcard"],
        picks=picks,
        history=pd.DataFrame(
            {"event": [5], "overall_points": [100], "overall_rank": [123], "event_transfers": [0],
             "active_chip": [None]}
        ),
        wildcard=ChipEvaluation("wildcard", 6, 12.5, "rebuild", plan=plan),
    )
    out = _my_team_payload(_report(mt), players)
    assert out is not None
    roles = {e["id"]: e["role"] for e in out["current"]}
    assert roles == {1: "XI", 2: "XI", 4: "vice", 5: "captain"}
    assert out["current"][1]["sell"] == 48  # sell price, not market price

    wc = out["wildcard"]
    assert wc["gain"] == 12.5 and wc["event"] == 6
    assert [(m["out"]["id"], m["in"]["id"]) for m in wc["moves"]] == [(2, 3)]
    assert wc["moves"][0]["delta_xp"] == 4.0  # (4-2) xP/GW over the 2-GW horizon
    assert {e["id"]: e["role"] for e in wc["squad"]}[5] == "captain"
    assert out["chips_used"] == [{"chip": "3xc", "event": 3}]
