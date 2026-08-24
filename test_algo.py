"""test_algo.py — invariants du design. `pytest test_algo.py -q`

Un seul invariant compte vraiment : AUCUNE sequence d'actions ramenant a un
etat deja visite ne peut avoir un retour cumule > 0. S'il tient, l'ordre
partiel "avancer > WAIT > osciller" est une consequence de l'arithmetique du
score, plus un equilibre fragile entre douze coefficients a retuner.
"""
from __future__ import annotations

import numpy as np
import pytest

from AlgoSpec import (
    HEROES, MAX_HEIGHT, MAX_WIDTH, N_GRID_CHANNELS, N_SCALARS, REWARD_CONFIG,
    WAIT_STREAK_LIMIT, MapSpec, format_actions_lines,
)
from AlgoEnv import AlgoEnv
from GameEngine import GameEngine


def spec(rows: list[str], elevation: list[list[int]], max_time: int = 100) -> MapSpec:
    elev = np.asarray(elevation, dtype=np.int8)
    elev.setflags(write=False)
    return MapSpec("test", tuple(rows), elev, max_time)


# M est mure dans son coin : son heuristique renvoie WAIT et n'interfere pas.
OPEN_MAP = spec(
    ["F.....",
     ".....#",
     "+...#M"],
    [[1, 1, 1, 1, 1, 1],
     [1, 1, 1, 1, 1, 0],
     [1, 1, 1, 1, 0, 1]],
)
BUSH_MAP = spec(["F*@..M"], [[1, 1, 1, 1, 1, 1]])
CLIFF_MAP = spec(["F*..M"], [[6, 6, 1, 1, 1]])
STONE_MAP = spec(["F+...M"], [[1, 1, 1, 1, 1, 1]])


def env(map_spec: MapSpec, hero: str = "F", **kw) -> AlgoEnv:
    e = AlgoEnv([map_spec], hero=hero, seed=0, **kw)
    e.reset()
    return e


def act(e: AlgoEnv, name: str):
    return e.step(e.action_names.index(name))


# --- score officiel --------------------------------------------------------
def test_official_score_matches_gdd_example():
    """Exemple chiffre du GDD : C=0, P=2, S=181, B=183, T=100 -> R=191."""
    e = GameEngine([".."], np.array([[1, 1]]), 100, seed=0)
    e.stones_collected, e.chests_hidden = 2, 0
    e.stamina = {"F": 81.0, "M": 100.0}
    e.battery = {"F": 93.0, "M": 90.0}
    e.tick = 0
    assert e.official_score() == 191


# --- le noyau : telescopage et absence de cycle positif -------------------
def test_reward_telescopes_to_local_score_delta():
    """Somme des recompenses == (score_local_final - initial)/25, exactement.
    Le bonus de nouveaute est neutralise pour isoler le telescopage."""
    e = env(OPEN_MAP, reward_config={"novelty_bonus": 0.0})
    start = e.local_score()
    total = 0.0
    rng = np.random.default_rng(7)
    for _ in range(40):
        legal = np.flatnonzero(e.action_masks())
        _, r, term, trunc, _ = e.step(int(rng.choice(legal)))
        total += r
        if term or trunc:
            break
    assert total == pytest.approx((e.local_score() - start) / 25.0, abs=1e-9)


def test_no_positive_cycle_wait():
    """Chaque WAIT est strictement negatif : -0.5 de temps contre au mieux
    +0.25 de regeneration (2 heros, 0.5/tick, /4) -> net -0.25 par tick."""
    e = env(OPEN_MAP)
    rewards = [act(e, "WAIT")[1] for _ in range(10)]
    assert all(r < 0 for r in rewards), rewards
    assert sum(rewards) < 0


def test_no_positive_cycle_round_trip():
    """Un aller-retour sur des cases deja visitees a un retour cumule < 0 :
    la nouveaute ne paie qu'une fois par case et par episode."""
    e = env(OPEN_MAP)
    act(e, "RIGHT")            # premiere visite : nouveaute consommee
    act(e, "LEFT")
    lap = act(e, "RIGHT")[1] + act(e, "LEFT")[1]
    assert lap < 0, lap
    second = act(e, "RIGHT")[1] + act(e, "LEFT")[1]
    assert second == pytest.approx(lap, abs=1e-6)   # pas de derive, pas de farm


def test_no_positive_cycle_blocked_move_spam():
    """Marteler un mur ne rapporte rien : implicit_wait paie le temps."""
    e = env(OPEN_MAP)
    rewards = [act(e, "UP")[1] for _ in range(5)]   # F est en (0,0), UP hors carte
    assert all(r < 0 for r in rewards), rewards


# --- echelle des trois evenements qui comptent ----------------------------
def test_stone_is_worth_about_one():
    _, r, _, _, info = act(env(STONE_MAP), "RIGHT")
    assert info["hero_stones"] == 1
    assert 0.9 < r < 1.1, r


def test_hidden_chest_is_worth_about_six():
    _, r, _, _, info = act(env(BUSH_MAP), "PUSH_RIGHT")
    assert info["hero_chests"] == 1
    assert 5.5 < r < 6.1, r


def test_destroyed_chest_costs_about_six():
    """Un coffre detruit ne fait pas baisser le score : il supprime un +150
    futur. Le -6.0 rend ce manque a gagner immediat."""
    _, r, _, _, info = act(env(CLIFF_MAP), "PUSH_RIGHT")
    assert info["hero_chests_destroyed"] == 1
    assert r < -5.9, r


# --- masque : contrat structurel, pas du shaping --------------------------
def test_action_mask_shape_and_never_empty():
    for hero in ("F", "M"):
        e = env(OPEN_MAP, hero=hero)
        mask = e.action_masks()
        assert mask.shape == (HEROES[hero]["n_actions"],)
        assert mask.any()


def test_wait_leaves_the_mask_after_streak():
    e = env(OPEN_MAP)
    wait = e.action_names.index("WAIT")
    for _ in range(WAIT_STREAK_LIMIT):
        assert e.action_masks()[wait]
        act(e, "WAIT")
    mask = e.action_masks()
    assert not mask[wait], "WAIT doit sortir du masque apres la serie"
    assert mask.any(), "une autre action legale doit rester disponible"


def test_moving_resets_the_wait_streak():
    e = env(OPEN_MAP)
    for _ in range(WAIT_STREAK_LIMIT):
        act(e, "WAIT")
    act(e, "RIGHT")
    assert e.action_masks()[e.action_names.index("WAIT")]


# --- observation ----------------------------------------------------------
def test_observation_contract():
    e = env(OPEN_MAP)
    obs, _, _, _, _ = act(e, "RIGHT")
    assert obs["grid"].shape == (N_GRID_CHANNELS, MAX_HEIGHT, MAX_WIDTH)
    assert obs["scalars"].shape == (N_SCALARS,)
    assert np.isfinite(obs["grid"]).all() and np.isfinite(obs["scalars"]).all()
    assert -1.0 <= obs["grid"].min() and obs["grid"].max() <= 1.0
    assert 0.0 <= obs["scalars"].min() and obs["scalars"].max() <= 1.0


# --- curriculum : distribution des etats de depart, rien d'autre ----------
def test_phase_zero_spawns_next_to_an_objective():
    e = AlgoEnv([OPEN_MAP], hero="F", curriculum_phase=0, seed=3)
    for _ in range(10):
        e.reset()
        hx, hy = e.engine.pos["F"]
        objectives = list(e.engine.stones) + [(c[0], c[1]) for c in e.engine.chests]
        d = min(abs(hx - ox) + abs(hy - oy) for ox, oy in objectives)
        assert 1 <= d <= 2, d


def test_last_phase_uses_gdd_positions():
    e = AlgoEnv([OPEN_MAP], hero="F", curriculum_phase=3, seed=3)
    e.reset()
    assert e.engine.pos["F"] == (0, 0)


# --- garde-fous anti-derive ----------------------------------------------
def test_reward_config_has_no_shaping_keys():
    """Toute cle supplementaire serait du shaping : c'est precisement ce que
    ce design refuse."""
    assert set(REWARD_CONFIG) == {"score_scale", "chest_destroyed", "novelty_bonus"}


def test_engine_exposes_only_the_legal_mask():
    assert not hasattr(GameEngine, "structural_action_mask")


def test_elevation_stays_readonly():
    e = env(OPEN_MAP)
    with pytest.raises(ValueError):
        e.map_spec.elevation[0, 0] = 9


def test_actions_file_format():
    lines = format_actions_lines([("LEFT", "LEFT"), ("HACK", "HACK_MOVE")])
    assert lines == ["LEFT | LEFT", "HACK | HACK_MOVE", "END_GAME"]
