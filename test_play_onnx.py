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
    """Strategie "Visite recente + WAIT modere" : les penalites WAIT doivent
    etre assouplies vs la 1re fix trop agressive pour que l'ordre partiel
    avancer (+1.05) > WAIT (-0.75) > osciller (-1.50 a -2.05) soit respecte.
    Si WAIT est trop fort (-1.50+), l'agent prefere oscillation (-0.05/pas)
    au lieu de WAIT, et le see-saw reapparait."""
    from AlgoEnv import _REWARD_DEFAULTS as rd
    # WAIT modere: -0.50 a -1.25 (pas -1.50+ comme la 1re fix).
    assert -1.25 <= rd["wait_productive_action_available"] <= -0.50, (
        f"wait_productive_action_available doit etre modere, "
        f"actuel: {rd['wait_productive_action_available']}"
    )
    assert -1.50 <= rd["wait_full_resources"] <= -0.75
    assert -1.50 <= rd["repeated_wait_3_plus"] <= -0.75
    # Penalite de revisite forte pour casser l'oscillation.
    assert rd["revisited_position"] <= -2.0


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


# ----------------------------------------------------------------------------
# Tests de la strategie "Visite recente + WAIT modere" (anti-balancier
# WAIT <-> oscillation)
# ----------------------------------------------------------------------------

def test_revisited_position_penalty_exists_and_strong():
    """La penalite revisited_position doit exister et etre suffisamment forte
    pour qu'un cycle d'oscillation (2-3-4) soit plus couteux que WAIT."""
    from AlgoEnv import _REWARD_DEFAULTS as rd
    assert rd["revisited_position"] <= -2.0, (
        "revisited_position doit etre <= -2.0 pour qu'une oscillation 2-cycle "
        f"ait un cout moyen <= WAIT. Actuel: {rd['revisited_position']}"
    )


def test_wait_penalty_softened():
    """La penalite wait_productive_action_available ne doit plus dominer les
    autres signaux : entre -0.50 et -1.00. Sinon le see-saw WAIT/oscillation
    reapparait (cf. 1re fix trop agressive)."""
    from AlgoEnv import _REWARD_DEFAULTS as rd
    assert -1.00 <= rd["wait_productive_action_available"] <= -0.50
    assert -1.50 <= rd["wait_full_resources"] <= -0.75
    assert -1.50 <= rd["repeated_wait_3_plus"] <= -0.75


def test_position_history_initialized_and_tracked():
    """AlgoEnv doit initialiser _position_history au reset et le mettre a
    jour a chaque deplacement (pas sur les WAIT)."""
    import os, tempfile
    import numpy as np
    from AlgoEnv import AlgoEnv
    from AlgoTrain import DEFAULT_ENGINE_CONFIG, DEFAULT_REWARD_CONFIG

    rows = [
        "F...M.",
        "......",
        "...*..",
        "...+..",
        "...@..",
    ]
    H, W = 5, 6
    elev = np.array([[1]*W for _ in range(H)], dtype=np.int64)
    tmp = tempfile.mkdtemp()
    map_path = os.path.join(tmp, "map.txt")
    elev_path = os.path.join(tmp, "elevation.txt")
    with open(map_path, "w") as f:
        f.write(f"{H} {W} 50\n")
        for r in rows:
            f.write(r + "\n")
    with open(elev_path, "w") as f:
        for row in elev:
            f.write(" ".join(map(str, row)) + "\n")

    env = AlgoEnv(map_path, elev_path, hero="F",
                  engine_config=DEFAULT_ENGINE_CONFIG,
                  reward_config=DEFAULT_REWARD_CONFIG, seed=42)
    env.reset()
    # Apres reset : l'historique contient la position de depart.
    assert len(env._position_history) == 1
    initial_pos = env.engine.pos["F"]
    assert tuple(initial_pos) in env._position_history

    # Trouve une direction legale.
    mask = env.engine.legal_action_mask("F", env.action_names)
    move_indices = [i for i, v in enumerate(mask)
                    if v and env.action_names[i] in ("UP", "DOWN", "LEFT", "RIGHT")]
    assert move_indices, "Pas de move legal au depart"
    a = move_indices[0]
    env.step(a)
    # Apres un deplacement : 2 positions dans l'historique.
    assert len(env._position_history) == 2

    # Si on WAIT, l'historique ne grandit pas (pas de pollution par WAIT).
    wait_idx = env.action_names.index("WAIT")
    env.step(wait_idx)
    assert len(env._position_history) == 2, "WAIT ne doit pas enrichir l'historique"


def test_oscillation_round_trip_costs_more_than_wait():
    """Test cible de la strategie : un aller-retour A->B->A doit couter PLUS
    que 2 WAIT consecutifs (avec action productive dispo), sinon le see-saw
    WAIT/oscillation reapparait.

    Note : on deplete stamina/battery avant le test pour simuler un scenario
    realiste ou wait_productive_action_available (-0.75) s'applique (et non
    wait_full_resources -1.00 qui est reserve au cas ressources pleines).
    """
    import os, tempfile
    import numpy as np
    from AlgoEnv import AlgoEnv
    from AlgoTrain import DEFAULT_ENGINE_CONFIG, DEFAULT_REWARD_CONFIG

    # Carte avec beaucoup d'espace pour autoriser un aller-retour simple.
    rows = [
        "F....M",
        "......",
        "......",
        "......",
        "......",
        "....+.",   # pierre loin pour avoir un signal directionnel stable
    ]
    H, W = 6, 6
    elev = np.array([[1]*W for _ in range(H)], dtype=np.int64)
    tmp = tempfile.mkdtemp()
    map_path = os.path.join(tmp, "map.txt")
    elev_path = os.path.join(tmp, "elevation.txt")
    with open(map_path, "w") as f:
        f.write(f"{H} {W} 100\n")
        for r in rows:
            f.write(r + "\n")
    with open(elev_path, "w") as f:
        for row in elev:
            f.write(" ".join(map(str, row)) + "\n")

    def _make_env():
        e = AlgoEnv(map_path, elev_path, hero="F",
                    engine_config=DEFAULT_ENGINE_CONFIG,
                    reward_config=DEFAULT_REWARD_CONFIG, seed=42)
        e.reset()
        # Simule un scenario realiste : ressources partiellement utilisees.
        # stamina=50 (pas >= 99.999), battery=50 -> wait_full_resources
        # ne s'applique pas, c'est wait_productive_action_available (-0.75).
        e.engine.stamina["F"] = 50.0
        e.engine.battery["F"] = 50.0
        # Reinitialise _prev_potential car la resource_term a change.
        e._prev_potential = e._potential()
        return e

    # --- Scenario A : aller-retour RIGHT puis LEFT ---
    env_a = _make_env()
    right_idx = env_a.action_names.index("RIGHT")
    left_idx = env_a.action_names.index("LEFT")
    _, r_right, _, _, _ = env_a.step(right_idx)
    _, r_left, _, _, _ = env_a.step(left_idx)
    osc_cost = r_right + r_left
    print(f"  Aller-retour RIGHT->LEFT: r_right={r_right:+.3f} r_left={r_left:+.3f} total={osc_cost:+.3f}")
    # Verifie que la penalite de revisite est bien appliquee sur le retour.
    # Attendu: r_left contient -2.0 de revisited_position.
    # r_left = -shaping(-0.50) - potential(-0.60) - step_time(-0.05) - revisit(-2.0) ≈ -3.15
    assert r_left < -2.0, (
        f"Le retour devrait declencher la penalite de revisite (-2.0), "
        f"r_left={r_left:+.3f} devrait etre < -2.0"
    )

    # --- Scenario B : 2 WAIT consecutifs ---
    env_b = _make_env()
    wait_idx = env_b.action_names.index("WAIT")
    _, w1, _, _, _ = env_b.step(wait_idx)
    _, w2, _, _, _ = env_b.step(wait_idx)
    wait_cost = w1 + w2
    print(f"  2x WAIT (ressources non pleines): w1={w1:+.3f} w2={w2:+.3f} total={wait_cost:+.3f}")

    # L'oscillation doit etre PLUS CHERE que 2 WAIT.
    assert osc_cost < wait_cost, (
        f"L'aller-retour ({osc_cost:+.3f}) devrait etre plus couteux que "
        f"2 WAIT ({wait_cost:+.3f}) — c'est le coeur de la strategie "
        f"anti-see-saw."
    )


def test_forward_step_is_positively_rewarded():
    """Un pas vers un objectif doit etre recompense positivement, pour que
    l'ordre partiel avancer > WAIT > osciller soit respecte."""
    import os, tempfile
    import numpy as np
    from AlgoEnv import AlgoEnv
    from AlgoTrain import DEFAULT_ENGINE_CONFIG, DEFAULT_REWARD_CONFIG

    # Pierre a 3 cases de F pour un signal directionnel clair sans collecte
    # immediate (sinon +25 ecrase le signal de shaping qu'on veut mesurer).
    rows = [
        "F..+M.",
        "......",
        "......",
        "....@.",
        "......",
    ]
    H, W = 5, 6
    elev = np.array([[1]*W for _ in range(H)], dtype=np.int64)
    tmp = tempfile.mkdtemp()
    map_path = os.path.join(tmp, "map.txt")
    elev_path = os.path.join(tmp, "elevation.txt")
    with open(map_path, "w") as f:
        f.write(f"{H} {W} 100\n")
        for r in rows:
            f.write(r + "\n")
    with open(elev_path, "w") as f:
        for row in elev:
            f.write(" ".join(map(str, row)) + "\n")

    env = AlgoEnv(map_path, elev_path, hero="F",
                  engine_config=DEFAULT_ENGINE_CONFIG,
                  reward_config=DEFAULT_REWARD_CONFIG, seed=42)
    env.reset()
    # F en (0,0), pierre en (3,0). RIGHT rapproche (Manhattan 3 -> 2).
    right_idx = env.action_names.index("RIGHT")
    _, r, _, _, info = env.step(right_idx)
    print(f"  RIGHT vers pierre (dist 3->2): r={r:+.3f} (potential+shaping-step_cost)")
    # Doit etre positif : potential +0.60 + shaping +0.50 - step_time 0.05 = ~+1.05.
    assert r > 0.5, f"Avancer vers la pierre devrait donner > +0.5, obtenu {r:+.3f}"
