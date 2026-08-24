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


# ----------------------------------------------------------------------------
# Tests de la strategie "Elevation-Aware BFS Potential"
# ----------------------------------------------------------------------------
# Cf. diagnostic : la distance de Manhattan ment a l'agent en presence de murs
# d'élévation. Un coffre "proche" en Manhattan peut etre physiquement
# inaccessible a pied. La BFS contrainte par l'élévation corrige ce mensonge
# et ajoute un proxy "machine" pour garder un gradient quand un objectif est
# inatteignable a pied.


def test_reward_configs_include_elevation_aware_keys():
    """Les 3 nouvelles cles (progress_to_machine, regress_from_machine,
    elevation_barrier_repeated) doivent exister dans _REWARD_DEFAULTS
    (AlgoEnv) ET DEFAULT_REWARD_CONFIG (AlgoTrain), avec valeurs alignees."""
    from AlgoEnv import _REWARD_DEFAULTS as env_rc
    from AlgoTrain import DEFAULT_REWARD_CONFIG as train_rc
    for key in ("progress_to_machine", "regress_from_machine",
                "elevation_barrier_repeated"):
        assert key in env_rc, f"{key} manquant dans AlgoEnv._REWARD_DEFAULTS"
        assert key in train_rc, f"{key} manquant dans AlgoTrain.DEFAULT_REWARD_CONFIG"
        assert env_rc[key] == train_rc[key], (
            f"dephasage sur {key}: env={env_rc[key]} train={train_rc[key]}"
        )
    # Penalite de mur d'élévation doit etre forte (>= 1.0 en valeur absolue).
    assert env_rc["elevation_barrier_repeated"] <= -1.0


def test_bfs_flat_map_matches_manhattan():
    """Sur terrain plat, la BFS doit donner les memes distances que Manhattan
    et couvrir toute la carte."""
    import os, tempfile
    import numpy as np
    from AlgoEnv import AlgoEnv
    from AlgoTrain import DEFAULT_ENGINE_CONFIG, DEFAULT_REWARD_CONFIG

    rows = [
        "F...M",
        ".....",
        ".....",
        "..+..",
        "..@..",
    ]
    H, W = 5, 5
    elev = np.ones((H, W), dtype=np.int64)
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

    bfs = env._elevation_bfs("F")
    # F en (0,0), pierre en (2,3) -> Manhattan = 5 = BFS sur terrain plat.
    assert bfs.get((2, 3)) == 5, f"BFS={bfs.get((2, 3))}, attendu 5"
    # La BFS couvre toute la carte (25 cases atteignables).
    assert len(bfs) == H * W, f"BFS couvre {len(bfs)} cases, attendu {H * W}"


def test_bfs_respects_elevation_wall():
    """Un mur d'élévation doit bloquer la BFS. Carte 1-ligne avec F, pierre
    juste derrière le mur, M a droite. F ne peut pas monter (diff 4 >= 3)."""
    import os, tempfile
    import numpy as np
    from AlgoEnv import AlgoEnv
    from AlgoTrain import DEFAULT_ENGINE_CONFIG, DEFAULT_REWARD_CONFIG

    rows = ["F+M"]   # H=1, W=3. F en (0,0), pierre en (1,0), M en (2,0).
    H, W = 1, 3
    # Elévation : [1, 5, 1] -> mur a (1,0) avec diff=4 >= 3 depuis (0,0).
    elev = np.array([[1, 5, 1]], dtype=np.int64)
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

    bfs = env._elevation_bfs("F")
    # F seul dans la BFS : (1,0) (pierre) et (2,0) (M) sont bloqués par le mur.
    assert (0, 0) in bfs and bfs[(0, 0)] == 0
    assert (1, 0) not in bfs, (
        f"(1,0) (pierre) devrait etre bloque par le mur (diff 4 >= 3), "
        f"BFS keys: {sorted(bfs.keys())}"
    )
    assert (2, 0) not in bfs, "(2,0) doit aussi etre inatteignable"


def test_bfs_on_machine_relaxes_elevation():
    """Sur machine, les contraintes deviennent |diff| < machine_height_block=5,
    donc le mur a diff=4 devient franchissable."""
    import os, tempfile
    import numpy as np
    from AlgoEnv import AlgoEnv
    from AlgoTrain import DEFAULT_ENGINE_CONFIG, DEFAULT_REWARD_CONFIG

    # 1 ligne, 4 colonnes : F en (0,0), X en (1,0), pierre en (2,0), M en (3,0).
    rows = ["FX+M"]
    H, W = 1, 4
    # Elévation : [1, 1, 5, 1] -> mur a (2,0) avec diff=4 >= 3 depuis (1,0).
    elev = np.array([[1, 1, 5, 1]], dtype=np.int64)
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

    # A pied depuis (0,0) : (1,0) atteignable (diff 0), (2,0) bloque (diff 4 >= 3).
    bfs_foot = env._elevation_bfs("F")
    assert (1, 0) in bfs_foot and bfs_foot[(1, 0)] == 1
    assert (2, 0) not in bfs_foot, (
        "A pied, la pierre doit etre inatteignable (diff 4 >= 3)"
    )

    # Simule le hack : F sur X, on_engine=0, machine hacked_by="F".
    env.engine.pos["F"] = (1, 0)
    env.engine.on_engine["F"] = 0
    env.engine.machines[0]["hacked_by"] = "F"

    # Sur machine : contraintes |diff| < 5. (2,0) a diff=4 depuis (1,0), OK.
    bfs_machine = env._elevation_bfs("F")
    assert (2, 0) in bfs_machine, (
        f"Sur machine, la pierre en (2,0) doit etre atteignable (diff 4 < 5). "
        f"BFS keys: {sorted(bfs_machine.keys())}"
    )
    assert bfs_machine[(2, 0)] == 1


def test_potential_uses_machine_proxy_when_objective_unreachable():
    """Quand un objectif est inatteignable a pied (mur d'élévation),
    _potential() doit utiliser un proxy = diamètre_carte + distance machine
    au lieu de retourner stone_term = 0 (comportement legacy qui n'envoyait
    aucun signal directionnel vers la machine)."""
    import os, tempfile
    import numpy as np
    from AlgoEnv import AlgoEnv
    from AlgoTrain import DEFAULT_ENGINE_CONFIG, DEFAULT_REWARD_CONFIG

    # 1 ligne, 4 colonnes : F en (0,0), X en (1,0), pierre en (2,0), M en (3,0).
    rows = ["FX+M"]
    H, W = 1, 4
    elev = np.array([[1, 1, 5, 1]], dtype=np.int64)
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

    # Pierre inatteignable a pied, machine X atteignable a 1 pas.
    bfs = env._get_bfs()
    assert env._bfs_nearest(bfs, env.engine.stones) is None, (
        "La pierre doit etre inatteignable a pied"
    )
    machine_d = env._bfs_nearest(bfs, env._machine_positions())
    assert machine_d == 1, f"Machine X a 1 pas de F, obtenu: {machine_d}"

    # Proxy pierre = MAP_DIAMETER + 1 = (1+4) + 1 = 6.
    # stone_term = -0.60 * 6 = -3.6 (sans proxy, serait 0.0).
    # Le potentiel doit etre suffisamment negatif pour que le proxy soit actif.
    pot = env._potential()
    # Seuil : avec proxy actif, pot < -2.5 (stone_term seul = -3.6).
    assert pot < -2.5, (
        f"Proxy doit rendre le potentiel bien negatif (stone_term = -3.6), "
        f"obtenu: {pot:.3f}"
    )


def test_elevation_barrier_repeated_penalty_fires():
    """3 implicit_wait/blocked_move consecutifs doivent declencher la
    penalite elevation_barrier_repeated (-1.50). Verifie aussi que le
    compteur se reinitialise sur une action non-bloquee."""
    import os, tempfile
    import numpy as np
    from AlgoEnv import AlgoEnv
    from AlgoTrain import DEFAULT_ENGINE_CONFIG, DEFAULT_REWARD_CONFIG

    # Carte simple sans murs (terrain plat) : on appelle _reward() directement
    # avec des evenements simules pour isoler la logique de streak.
    rows = ["F..M"]
    H, W = 1, 4
    elev = np.ones((H, W), dtype=np.int64)
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
    env._elevation_block_streak = 0

    blocked_event = {"valid": True, "kind": "implicit_wait",
                     "reason": "blocked_move"}
    other_ev = {"valid": True, "kind": "wait"}

    # 1er blocked_move : streak -> 1, pas de penalite.
    r1 = env._reward(
        my=blocked_event, other_ev=other_ev,
        requested_action="RIGHT", selected_is_legal=True,
        before_chests=set(), before_hidden=0, before_destroyed=0,
        before_battery=100.0, before_stamina=100.0,
        productive_action_available=True, revisited=False,
    )
    assert env._elevation_block_streak == 1
    # Pas de penalite (-1.50) sur le 1er : r1 = step_time_cost (-0.05) seul.
    assert r1 > -1.0, f"1er blocked_move ne doit pas penaliser, r1={r1:.3f}"

    # 2e blocked_move : streak -> 2, pas de penalite.
    r2 = env._reward(
        my=blocked_event, other_ev=other_ev,
        requested_action="RIGHT", selected_is_legal=True,
        before_chests=set(), before_hidden=0, before_destroyed=0,
        before_battery=100.0, before_stamina=100.0,
        productive_action_available=True, revisited=False,
    )
    assert env._elevation_block_streak == 2
    assert r2 > -1.0, f"2e blocked_move ne doit pas penaliser, r2={r2:.3f}"

    # 3e blocked_move : streak -> 3, penalite -1.50 declenchee.
    r3 = env._reward(
        my=blocked_event, other_ev=other_ev,
        requested_action="RIGHT", selected_is_legal=True,
        before_chests=set(), before_hidden=0, before_destroyed=0,
        before_battery=100.0, before_stamina=100.0,
        productive_action_available=True, revisited=False,
    )
    assert env._elevation_block_streak == 3
    assert r3 <= -1.0, (
        f"3e blocked_move doit declencher -1.50, r3={r3:.3f}"
    )

    # Une action non-bloquee doit reinitialiser le streak.
    move_event = {"valid": True, "kind": "move", "cost": 1.0}
    env._reward(
        my=move_event, other_ev=other_ev,
        requested_action="RIGHT", selected_is_legal=True,
        before_chests=set(), before_hidden=0, before_destroyed=0,
        before_battery=100.0, before_stamina=100.0,
        productive_action_available=True, revisited=False,
    )
    assert env._elevation_block_streak == 0, "Action non-bloquee doit reset le streak"


def test_step_does_not_crash_with_elevation_walls():
    """Smoke test : un env avec murs d'élévation doit pouvoir tourner
    20 steps sans erreur, et la BFS doit etre invalidee a chaque step."""
    import os, tempfile
    import numpy as np
    from AlgoEnv import AlgoEnv
    from AlgoTrain import DEFAULT_ENGINE_CONFIG, DEFAULT_REWARD_CONFIG

    rows = [
        "F+..M",   # row 0 : F en (0,0), pierre en (1,0), M en (4,0)
        ".....",   # row 1
        ".....",   # row 2
        "X...G",   # row 3 : X en (0,3), G en (4,3)
        ".....",   # row 4
    ]
    H, W = 5, 5
    # Mur d'élévation a (1,0) : la pierre y est posée mais inaccessible a pied
    # car F doit monter de 1 a 5 (diff 4 >= 3).
    elev = np.array([
        [1, 5, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
    ], dtype=np.int64)
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
    # Au reset : la pierre en (1,0) est inatteignable a pied (mur d'élévation).
    # La BFS doit le refléter.
    bfs = env._get_bfs()
    assert (1, 0) not in bfs, (
        f"Pierre en (1,0) doit etre inatteignable a pied (mur). "
        f"BFS keys: {sorted(bfs.keys())}"
    )
    # Le proxy machine doit etre actif dans _potential (X atteignable a 3 pas
    # via colonne 0).
    assert env._potential() < -2.0

    # 20 steps : ne doit pas crasher, la BFS doit etre reinvalidee a chaque step.
    for i in range(20):
        mask = env.engine.legal_action_mask("F", env.action_names)
        legal = [i for i, v in enumerate(mask) if v]
        if not legal:
            break
        a = legal[i % len(legal)]
        obs, r, term, trunc, info = env.step(a)
        # La BFS cachee doit etre reinvalidee a chaque step puis recalculee
        # paresseusement. Apres step, _cached_bfs peut etre None (invalide
        # avant mutation) ou rempli (si _reward a deja ete appele).
        # Verifie qu'apres _get_bfs(), le cache est rempli.
        _ = env._get_bfs()
        assert env._cached_bfs is not None
        if term or trunc:
            break

