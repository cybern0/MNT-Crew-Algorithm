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


# ----------------------------------------------------------------------------
# Tests des correctifs apportes au diagnostic (causes 1, 3, 4)
# ----------------------------------------------------------------------------

def test_reward_defaults_symmetric_shaping():
    """Cause 3 : progress et |regress| doivent etre symetriques pour qu'un
    aller-retour ait un gain net nul et ne declenche pas d'oscillation."""
    from AlgoEnv import _REWARD_DEFAULTS as rd
    assert rd["progress_to_stone"] == -rd["regress_from_stone"]
    assert rd["progress_to_chest"] == -rd["regress_from_chest"]


def test_reward_defaults_wait_penalties_reinforced():
    """Cause 1+4 : les penalites WAIT doivent etre assez fortes pour que
    l'agent n'apprenne jamais WAIT quand une action productive est legale."""
    from AlgoEnv import _REWARD_DEFAULTS as rd
    assert rd["wait_productive_action_available"] <= -1.50
    assert rd["wait_full_resources"] <= -2.00
    assert rd["repeated_wait_3_plus"] <= -2.00


def test_potential_coefficients_reinforced():
    """Cause 1 : le potentiel doit etre >= 3x le cout par pas (0.05) afin
    que le gain net du shaping telescoping surpasse la penalite d'action
    invalide (-0.25)."""
    from AlgoEnv import AlgoEnv
    # On inspecte le source : les coefficients doivent etre -0.60 / -0.75.
    import inspect
    src = inspect.getsource(AlgoEnv._potential)
    assert "-0.60" in src
    assert "-0.75" in src


def test_algoenv_algo_train_reward_configs_aligned():
    """Le reward_config de l'env et celui de l'entrainement doivent etre
    strictement alignes pour eviter un dephasage de signal."""
    from AlgoEnv import _REWARD_DEFAULTS as env_rc
    from AlgoTrain import DEFAULT_REWARD_CONFIG as train_rc
    common = set(env_rc) & set(train_rc)
    for k in common:
        assert env_rc[k] == train_rc[k], f"dephasage sur {k}: env={env_rc[k]} train={train_rc[k]}"


def test_has_productive_action_includes_push_and_hack():
    """Cause 4 : _has_productive_action doit reconnaitre PUSH_* et HACK hors
    engine comme productifs, sinon la penalite wait_productive_action_available
    tombe a wait_no_productive_action et l'agent apprend a WAIT."""
    from AlgoEnv import AlgoEnv
    import inspect
    src = inspect.getsource(AlgoEnv._has_productive_action)
    assert "PUSH_UP" in src and "PUSH_DOWN" in src
    assert "HACK_MOVE" in src and "HACK_FILL" in src


def test_playonnx_uses_legal_action_mask_not_structural():
    """Cause 2 : PlayOnnx.main() doit appeler legal_action_mask (precis),
    pas structural_action_mask (qui autorise des actions converties en
    implicit_wait par le moteur -> politique degénérée WAIT)."""
    import inspect, PlayOnnx
    src = inspect.getsource(PlayOnnx.main)
    code_lines = [l for l in src.splitlines()
                  if "legal_action_mask" in l and not l.strip().startswith("#")]
    assert code_lines, "PlayOnnx.main n'appelle pas legal_action_mask"
    structural_lines = [l for l in src.splitlines()
                        if "structural_action_mask" in l and not l.strip().startswith("#")]
    assert not structural_lines, "PlayOnnx.main utilise encore structural_action_mask"


def test_playonnx_has_anti_stuck_logic():
    """Cause 4 / boucle WAIT infinie : apres STUCK_THRESHOLD WAIT consecutifs,
    PlayOnnx doit forcer une action hors-WAIT tiree au hasard."""
    import inspect, PlayOnnx
    src = inspect.getsource(PlayOnnx.main)
    assert "STUCK_THRESHOLD" in src
    assert "consecutive_waits" in src
    assert "random.choice" in src
