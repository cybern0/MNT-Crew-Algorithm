"""test_play_onnx.py — Tests de non-regression pour PlayOnnx.py.

Execution :
    pytest test_play_onnx.py -q
"""
import numpy as np

from AlgoTrain import HEROES, action_mask
from GameEngine import GameEngine
from PlayOnnx import format_actions_lines


def _tiny_engine(max_time: int = 100) -> GameEngine:
    rows = [".."]
    elevation = np.array([[1, 1]])
    return GameEngine(rows, elevation, max_time, seed=0)


def test_official_score_matches_gdd_example():
    """Reprend l'exemple chiffre du GDD (section Scoring) :
    C=0, P=2, S=181, B=183, T=100 -> R=191.
    R = (C*150) + (P*25) + ((S+B)//4) + (T//2)
    """
    e = _tiny_engine(max_time=100)
    e.stones_collected = 2
    e.chests_hidden = 0
    e.stamina = {"F": 81.0, "M": 100.0}   # somme = 181
    e.battery = {"F": 93.0, "M": 90.0}    # somme = 183
    e.tick = 0                             # T = max_time - tick = 100

    assert e.official_score() == 191


def test_official_score_counts_hidden_chests_and_remaining_time():
    e = _tiny_engine(max_time=50)
    e.stones_collected = 0
    e.chests_hidden = 1
    e.stamina = {"F": 0.0, "M": 0.0}
    e.battery = {"F": 0.0, "M": 0.0}
    e.tick = 10  # T = 50 - 10 = 40

    # R = (1*150) + 0 + 0 + (40 // 2) = 170
    assert e.official_score() == 170


def test_action_mask_off_engine_allows_move_wait_push_hack_only():
    for hero in ("F", "M"):
        names = HEROES[hero]["actions"]
        mask = action_mask(is_on_engine=False, action_names=names)
        for name, valid in zip(names, mask):
            expected = name in (
                "UP", "DOWN", "LEFT", "RIGHT", "WAIT",
                "PUSH_UP", "PUSH_DOWN", "PUSH_LEFT", "PUSH_RIGHT", "HACK",
            )
            assert bool(valid) == expected, f"{hero}/{name}: attendu {expected}, obtenu {valid}"


def test_action_mask_on_engine_allows_wait_and_hack_star_only():
    for hero in ("F", "M"):
        names = HEROES[hero]["actions"]
        mask = action_mask(is_on_engine=True, action_names=names)
        for name, valid in zip(names, mask):
            expected = name == "WAIT" or name.startswith("HACK_")
            assert bool(valid) == expected, f"{hero}/{name}: attendu {expected}, obtenu {valid}"


def test_action_mask_always_has_at_least_one_valid_action():
    for hero in ("F", "M"):
        names = HEROES[hero]["actions"]
        for on_engine in (True, False):
            mask = action_mask(is_on_engine=on_engine, action_names=names)
            assert mask.any(), f"aucune action valide pour {hero} (on_engine={on_engine})"


def test_format_actions_lines_matches_gdd_format():
    pairs = [("LEFT", "LEFT"), ("HACK", "HACK_MOVE")]
    lines = format_actions_lines(pairs)

    assert lines == ["LEFT | LEFT", "HACK | HACK_MOVE", "END_GAME"]
    assert lines[-1] == "END_GAME"
    for line in lines[:-1]:
        left, right = line.split(" | ")
        assert left and right
