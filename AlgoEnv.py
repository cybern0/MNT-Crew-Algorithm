"""AlgoEnv.py — environnement gymnasium : observation 15 canaux, recompense a
deux termes, masque d'actions legal, reverse curriculum.

Recompense (par tick, pour le hero controle h) :

    r = [ 150*dC_h + 25*dP_h + (dS_h + dB_h)/4 + dT/2 ] / 25
        - 6 * dD_h
        + beta si la case atteinte est neuve dans cet episode

La somme des recompenses sur un episode vaut exactement
(score_local_final - score_local_initial)/25 + beta * (cases visitees).
Le telescopage est structurel : la politique optimale est identiquement celle
qui maximise le score du GDD. Aucun cycle n'est positif (cf. test_algo.py).

L'exploration ne passe PAS par la recompense mais par :
  - le reverse curriculum (distribution des etats de depart, ne touche pas
    a l'optimalite) ;
  - le retrait de WAIT du masque apres WAIT_STREAK_LIMIT WAIT consecutifs
    (contrainte structurelle deterministe, zero impact sur r).
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from AlgoSpec import (
    ABS_ELEV_CH, CURRICULUM_PHASES, FACING_CH, HERO_CH, HEROES,
    LOOKAHEAD_ONLY_IF_CHEST_AHEAD, LOOKAHEAD_TICKS, MAX_HEIGHT, MAX_WIDTH,
    NEXT_G_CH, NEXT_X_CH, N_GRID_CHANNELS, N_SCALARS, REL_ELEV_CH,
    RESOURCE_LOW_TICKS_LIMIT, REWARD_CONFIG, TILE_CHANNELS, WAIT_STREAK_LIMIT,
    MapSpec, load_maps,
)
from GameEngine import GameEngine


# --- encodage de l'observation (contrat ONNX/C#, reutilise par PlayOnnx) ----
def build_grid(engine: GameEngine, hero: str, elevation: np.ndarray) -> np.ndarray:
    g = np.zeros((N_GRID_CHANNELS, MAX_HEIGHT, MAX_WIDTH), dtype=np.float32)
    e = engine
    for y in range(e.H):
        row = e.terrain[y]
        for x in range(e.W):
            ch = TILE_CHANNELS.get(row[x])
            if ch is not None:
                g[ch, y, x] = 1.0
    for cx, cy in e.chests:
        g[TILE_CHANNELS["*"], cy, cx] = 1.0
    for sx, sy in e.stones:
        g[TILE_CHANNELS["+"], sy, sx] = 1.0
    for m in e.machines:
        g[TILE_CHANNELS[m["type"]], m["y"], m["x"]] = 1.0

    hx, hy = e.pos[hero]
    g[HERO_CH, hy, hx] = 1.0
    fdx, fdy = e.facing[hero]
    fx, fy = hx + fdx, hy + fdy
    facing_ok = e.in_bounds(fx, fy)
    if facing_ok:
        g[FACING_CH, fy, fx] = 1.0

    elev = elevation.astype(np.float32)
    g[ABS_ELEV_CH, :e.H, :e.W] = (elev / 4.5) - 1.0
    g[REL_ELEV_CH, :e.H, :e.W] = np.clip((elev - float(elev.mean())) / 4.5, -1.0, 1.0)

    chest_ahead = facing_ok and e.chest_at(fx, fy) is not None
    if chest_ahead or not LOOKAHEAD_ONLY_IF_CHEST_AHEAD:
        preview = e.preview_machine_positions(LOOKAHEAD_TICKS)
        for px, py in preview["X"]:
            g[NEXT_X_CH, py, px] = 1.0
        for px, py in preview["G"]:
            g[NEXT_G_CH, py, px] = 1.0
    return g


def build_scalars(engine: GameEngine, hero: str) -> np.ndarray:
    e = engine
    hx, hy = e.pos[hero]
    s = np.zeros((N_SCALARS,), dtype=np.float32)
    s[0] = np.clip(e.stamina[hero] / 100.0, 0.0, 1.0)
    s[1] = np.clip(e.battery[hero] / 100.0, 0.0, 1.0)
    s[2] = np.clip((e.max_time - e.tick) / e.max_time, 0.0, 1.0) if e.max_time > 0 else 0.0
    s[3] = np.clip(hx / max(1, e.W - 1), 0.0, 1.0)
    s[4] = np.clip(hy / max(1, e.H - 1), 0.0, 1.0)
    s[5] = 1.0 if e.on_engine[hero] is not None else 0.0
    return s


class AlgoEnv(gym.Env):
    """Une instance = un hero controle. L'autre suit GameEngine.scripted_action()."""

    metadata = {"render_modes": []}

    def __init__(self, maps: list[MapSpec], hero: str = "F",
                 reward_config: dict | None = None,
                 curriculum_phase: int | None = None, seed: int = 0):
        super().__init__()
        if hero not in HEROES:
            raise ValueError(f"hero inconnu : {hero!r}")
        if not maps:
            raise ValueError("au moins une carte est requise")
        self.hero = hero
        self.other = "M" if hero == "F" else "F"
        self.action_names = HEROES[hero]["actions"]
        self.target_machine = HEROES[hero]["machine"]
        self.maps = list(maps)
        self.rc = {**REWARD_CONFIG, **(reward_config or {})}
        # Phase par defaut = derniere (positions du GDD) : un env cree sans
        # curriculum se comporte comme l'environnement d'evaluation.
        self.phase = len(CURRICULUM_PHASES) - 1 if curriculum_phase is None else curriculum_phase

        self.observation_space = spaces.Dict({
            "grid": spaces.Box(-1.0, 1.0, (N_GRID_CHANNELS, MAX_HEIGHT, MAX_WIDTH), np.float32),
            "scalars": spaces.Box(0.0, 1.0, (N_SCALARS,), np.float32),
        })
        self.action_space = spaces.Discrete(HEROES[hero]["n_actions"])

        self._rng = np.random.default_rng(seed)
        self.engine: GameEngine | None = None
        self.map_spec: MapSpec | None = None
        self._prev_score = 0.0
        self._visited: set[tuple[int, int]] = set()
        self._wait_streak = 0

    # ---- curriculum -------------------------------------------------------
    def set_curriculum_phase(self, phase: int) -> None:
        self.phase = int(np.clip(phase, 0, len(CURRICULUM_PHASES) - 1))

    def _spawn_candidates(self) -> list[tuple[int, int]]:
        """Etats de depart de la phase courante. Ne touche qu'a reset(), donc
        ne peut pas deformer la politique optimale."""
        e = self.engine
        band = CURRICULUM_PHASES[self.phase]
        if band is None:
            return []
        if band == "machine":
            return [(m["x"], m["y"]) for m in e.machines
                    if m["type"] == self.target_machine]
        objectives = list(e.stones) + [(cx, cy) for cx, cy in e.chests]
        if not objectives:
            return []
        lo, hi = band
        out = []
        for y in range(e.H):
            for x in range(e.W):
                if not e.can_stand(self.hero, x, y):
                    continue
                d = min(abs(x - ox) + abs(y - oy) for ox, oy in objectives)
                if lo <= d <= hi:
                    out.append((x, y))
        return out

    # ---- masque -----------------------------------------------------------
    def action_masks(self) -> np.ndarray:
        """Contrat MaskablePPO. Masque legal exact, moins WAIT si le hero
        vient d'enchainer WAIT_STREAK_LIMIT attentes alors qu'autre chose est
        possible. Deterministe, non hackable, hors recompense."""
        mask = self.engine.legal_action_mask(self.hero, self.action_names).copy()
        wait = self.action_names.index("WAIT")
        if self._wait_streak >= WAIT_STREAK_LIMIT and mask[wait] and mask.sum() > 1:
            mask[wait] = False
        return mask

    # ---- score local (base de la recompense) ------------------------------
    def local_score(self) -> float:
        """Score officiel restreint a la contribution du hero controle, en
        float (pas de // : les zones mortes de quantification effaceraient
        des progres reels de stamina/temps)."""
        e, h = self.engine, self.hero
        return (150.0 * e.chests_hidden_by[h]
                + 25.0 * e.stones_by[h]
                + 0.25 * (e.stamina[h] + e.battery[h])
                + 0.5 * max(0, e.max_time - e.tick))

    # ---- gym API ----------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.map_spec = self.maps[int(self._rng.integers(len(self.maps)))]
        self.engine = GameEngine(
            list(self.map_spec.rows), self.map_spec.elevation, self.map_spec.max_time,
            seed=int(self._rng.integers(0, 2 ** 31 - 1)),
        )
        candidates = self._spawn_candidates()
        if candidates:
            self.engine.pos[self.hero] = candidates[int(self._rng.integers(len(candidates)))]
        self._prev_score = self.local_score()
        self._visited = {tuple(self.engine.pos[self.hero])}
        self._wait_streak = 0
        return self._obs(), self._info()

    def step(self, action):
        e, h = self.engine, self.hero
        a_name = self.action_names[int(action)]
        before_destroyed = e.chests_destroyed_by[h]

        ev = e.step({h: a_name, self.other: e.scripted_action(self.other)})
        my = ev[h]

        # --- Terme 1 : difference de score officiel (telescopique) ---
        score = self.local_score()
        reward = (score - self._prev_score) / self.rc["score_scale"]
        self._prev_score = score
        # --- Terme 2 : cout d'opportunite d'un coffre detruit ---
        reward += self.rc["chest_destroyed"] * (e.chests_destroyed_by[h] - before_destroyed)
        # --- Bonus de nouveauté, une fois par case et par episode ---
        pos = tuple(e.pos[h])
        if pos not in self._visited:
            self._visited.add(pos)
            reward += self.rc["novelty_bonus"]

        idle = a_name == "WAIT" or my["kind"] in ("implicit_wait", "wait")
        self._wait_streak = self._wait_streak + 1 if idle else 0

        cleared = not e.stones and not e.chests
        exhausted = e.resource_low_ticks[h] >= RESOURCE_LOW_TICKS_LIMIT
        terminated = bool(cleared or exhausted)
        truncated = bool(not terminated and e.tick >= e.max_time)

        info = self._info()
        info["action_name"] = a_name
        info["action_result"] = my["kind"]
        if terminated or truncated:
            info["termination_reason"] = ("objectives_cleared" if cleared
                                          else "resources_exhausted" if exhausted
                                          else "timeout")
        return self._obs(), float(reward), terminated, truncated, info

    def _obs(self):
        return {"grid": build_grid(self.engine, self.hero, self.map_spec.elevation),
                "scalars": build_scalars(self.engine, self.hero)}

    def _info(self):
        e, h = self.engine, self.hero
        return {
            "map": self.map_spec.name,
            "phase": self.phase,
            "is_on_engine": e.on_engine[h] is not None,
            "hero_pos": tuple(e.pos[h]),
            "hero_stones": e.stones_by[h],
            "hero_chests": e.chests_hidden_by[h],
            "hero_chests_destroyed": e.chests_destroyed_by[h],
            "stones_collected": e.stones_collected,
            "chests_hidden": e.chests_hidden,
            "chests_destroyed": e.chests_destroyed,
            "official_score": e.official_score(),
            "local_score": self.local_score(),
            "tick": e.tick,
            "action_mask": self.action_masks(),
        }


def make_env(map_paths, elevation_paths, hero="F", **kwargs) -> AlgoEnv:
    """Point d'entree pratique : chemins de fichiers -> env pret."""
    return AlgoEnv(load_maps(list(map_paths), list(elevation_paths)), hero=hero, **kwargs)
