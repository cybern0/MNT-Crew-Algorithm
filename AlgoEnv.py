"""AlgoEnv.py — Environnement gymnasium reel pour AlgoGames 2 (GDD + Twist).

Encodage 15 canaux IDENTIQUE au contrat documente dans AlgoTrain.AlgoGamesEnv
(meme mapping de tuiles, memes formules de normalisation d'elevation), mais
anime par GameEngine.py : vraies regles (F/M/X/G, hop/climb, push+Twist,
hacking, autopilote machines, look-ahead collisions) au lieu du squelette.

Charge dynamiquement par AlgoTrain.main() via find_factory() -> make_env().
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from AlgoTrain import (
    HEROES, MAX_HEIGHT, MAX_WIDTH, N_GRID_CHANNELS, N_SCALARS, AUGMENTATIONS,
    LOOKAHEAD_TICKS, action_mask, read_map_header, read_elevation, validate_terrain,
)
from GameEngine import GameEngine

_TILE_CHANNELS = {".": 0, "#": 1, "*": 2, "o": 3, "t": 4, "@": 5, "+": 6, "X": 7, "G": 8}
_HERO_CH, _FACING_CH, _ABS_ELEV_CH, _REL_ELEV_CH, _NEXT_X_CH, _NEXT_G_CH = 9, 10, 11, 12, 13, 14

_STREAK_LIMIT = 5
_IDLE_STREAK_LIMIT = 15  # au dela, WAIT continue d'etre penalise en plus de idle_action
_RESOURCE_LOW_TICKS_LIMIT = 10
_SCORE_BONUS_SCALE = 0.02  # bonus terminal = score officiel GDD * scale

# ---------------------------------------------------------------------------
# REDESIGN RECOMPENSE (cf. diagnostic "politique degenere en WAIT permanent") :
#
# Constat empirique : sur 10k timesteps, les 30 trials Optuna convergeaient
# TOUS vers le meme score (au bit pres), quels que soient les hyperparametres
# testes -> preuve que la politique tombe sur un point fixe trivial (WAIT
# partout) independant des hyperparametres. Cause racine, dans l'ancien
# bareme :
#   1. Un `move` reussi qui ne ramasse pas de pierre ne rapportait RIEN
#      d'autre que `progress_to_objective` (poids 0.05) -> seul signal pour
#      inciter a bouger, et il est minuscule.
#   2. `voluntary_chest_loss` (-75) valait ~3750x le cout d'un WAIT (-0.02) :
#      une seule maladresse pendant l'exploration aleatoire initiale suffit
#      a rendre l'esperance de toute action risquee tres negative.
#   3. WAIT n'avait aucun cout croissant : attendre indefiniment restait la
#      strategie la moins pire, pour toujours.
#
# Correctifs (les 3 sont complementaires, aucun ne suffit seul) :
#   - progress_to_objective x5 (0.05 -> 0.25) : signal dense a chaque tick,
#     domine desormais le bruit d'une pénalité isolee sur un episode de
#     500 ticks (jusqu'a +125 cumules si l'agent progresse constamment).
#   - voluntary_chest_loss / chest_destroyed largement reduits (-75 -> -20,
#     -35 -> -10) : reste clairement dissuasif mais n'ecrase plus a lui
#     seul des dizaines de ticks de bon comportement dans l'estimation
#     d'avantage de PPO.
#   - prolonged_block adouci (-1.00 -> -0.50) : les series d'actions
#     invalides pendant l'exploration aleatoire du debut ne doivent pas
#     etre punies aussi durement qu'une perte de coffre volontaire.
#   - idle_streak_penalty (nouveau) : au-dela de _IDLE_STREAK_LIMIT WAIT
#     consecutifs, chaque WAIT supplementaire coute idle_action +
#     idle_streak_penalty. Un episode 100% WAIT (500 ticks) coute desormais
#     15*(-0.02) + 485*(-0.02-0.10) ~= -58.5, largement pire qu'une seule
#     maladresse (-20), donc WAIT-permanent n'est plus un optimum local
#     "sur" -- l'agent est force a explorer.
# ---------------------------------------------------------------------------
_REWARD_DEFAULTS = dict(
    stone_collected=25.0, chest_hidden=150.0,
    strategic_action=0.30, useful_hack=0.75, useful_fill=1.00, useful_cut=1.00,
    useful_push=0.50, progress_to_objective=0.25,
    invalid_action=-0.15, idle_action=-0.02, prolonged_block=-0.50,
    idle_streak_penalty=-0.10,
    chest_destroyed=-10.0, voluntary_chest_loss=-20.0,
    resource_exhausted=-5.0, timeout=0.0,
)


def _augment(rows, elevation, mode):
    m = np.array([list(r) for r in rows])
    e = elevation
    if mode == "transpose":
        m, e = m.T, e.T
    elif mode == "rotate90":
        m, e = np.rot90(m), np.rot90(e)
    elif mode == "rotate180":
        m, e = np.rot90(m, 2), np.rot90(e, 2)
    elif mode == "rotate270":
        m, e = np.rot90(m, 3), np.rot90(e, 3)
    elif mode == "mirror_horizontal":
        m, e = np.fliplr(m), np.fliplr(e)
    elif mode == "mirror_vertical":
        m, e = np.flipud(m), np.flipud(e)
    return ["".join(r) for r in m], np.ascontiguousarray(e, dtype=elevation.dtype)


def build_grid(engine: GameEngine, hero: str, elevation: np.ndarray,
                engine_config: dict, grid_channels: int = N_GRID_CHANNELS) -> np.ndarray:
    """Encodage 15 canaux (contrat AlgoGamesEnv/ONNX). Fonction libre pour
    pouvoir etre appelee hors gym.Env (ex: script d'inference ONNX)."""
    g = np.zeros((grid_channels, MAX_HEIGHT, MAX_WIDTH), dtype=np.float32)
    e = engine
    for y in range(e.H):
        row = e.terrain[y]
        for x in range(e.W):
            ch = _TILE_CHANNELS.get(row[x])
            if ch is not None:
                g[ch, y, x] = 1.0
    for cx, cy in e.chests:
        g[_TILE_CHANNELS["*"], cy, cx] = 1.0
    for sx, sy in e.stones:
        g[_TILE_CHANNELS["+"], sy, sx] = 1.0
    for m in e.machines:
        g[_TILE_CHANNELS[m["type"]], m["y"], m["x"]] = 1.0

    hx, hy = e.pos[hero]
    g[_HERO_CH, hy, hx] = 1.0
    fdx, fdy = e.facing[hero]
    fx, fy = hx + fdx, hy + fdy
    facing_in_bounds = e.in_bounds(fx, fy)
    if facing_in_bounds:
        g[_FACING_CH, fy, fx] = 1.0

    elev = elevation.astype(np.float32)
    g[_ABS_ELEV_CH, :e.H, :e.W] = (elev / 4.5) - 1.0
    mean_elev = float(elev.mean())
    g[_REL_ELEV_CH, :e.H, :e.W] = np.clip((elev - mean_elev) / 4.5, -1.0, 1.0)

    if engine_config.get("look_ahead", True):
        only_if_chest = engine_config.get("lookahead_only_if_chest_ahead", True)
        chest_ahead = facing_in_bounds and e.chest_at(fx, fy) is not None
        if chest_ahead or not only_if_chest:
            preview = e.preview_machine_positions(LOOKAHEAD_TICKS)
            for px, py in preview["X"]:
                g[_NEXT_X_CH, py, px] = 1.0
            for px, py in preview["G"]:
                g[_NEXT_G_CH, py, px] = 1.0
    return g


def build_scalars(engine: GameEngine, hero: str, scalar_count: int = N_SCALARS) -> np.ndarray:
    e = engine
    s = np.zeros((scalar_count,), dtype=np.float32)
    hx, hy = e.pos[hero]
    s[0] = np.clip(e.stamina[hero] / 100.0, 0.0, 1.0)
    s[1] = np.clip(e.battery[hero] / 100.0, 0.0, 1.0)
    s[2] = np.clip((e.max_time - e.tick) / e.max_time, 0.0, 1.0) if e.max_time > 0 else 0.0
    s[3] = np.clip(hx / max(1, e.W - 1), 0.0, 1.0)
    s[4] = np.clip(hy / max(1, e.H - 1), 0.0, 1.0)
    s[5] = 1.0 if e.on_engine[hero] is not None else 0.0
    return s


class AlgoEnv(gym.Env):
    """1 instance = 1 hero controle ('F' ou 'M'). L'autre hero est pilote par
    GameEngine.scripted_action() (heuristique simple, sujette au meme
    look-ahead de collisions que tout le reste)."""

    metadata = {"render_modes": []}

    def __init__(self, map_path, elevation_path, hero="F", augmentation="identity",
                 augmentations=AUGMENTATIONS, engine_config=None, reward_config=None,
                 seed=0, max_height=MAX_HEIGHT, max_width=MAX_WIDTH,
                 grid_channels=N_GRID_CHANNELS, scalar_count=N_SCALARS):
        super().__init__()
        if hero not in HEROES:
            raise ValueError(f"hero inconnu : {hero!r}. Attendu : {list(HEROES)}.")
        self.hero = hero
        self.other = "M" if hero == "F" else "F"
        self.action_names = HEROES[hero]["actions"]
        self.n_actions = len(self.action_names)
        self.engine_config = dict(engine_config or {})
        self.reward_config = {**_REWARD_DEFAULTS, **(reward_config or {})}
        self.grid_channels, self.scalar_count = grid_channels, scalar_count

        self.map_path, self.elevation_path = Path(map_path), Path(elevation_path)
        h, w, self.max_time, base_rows = read_map_header(self.map_path)
        base_elev = read_elevation(self.elevation_path, h, w)
        validate_terrain(base_rows, base_elev)
        mode = augmentation if augmentation in augmentations else "identity"
        self.ascii_rows, self.elevation = _augment(base_rows, base_elev, mode)
        self.height, self.width = len(self.ascii_rows), len(self.ascii_rows[0])

        self.observation_space = spaces.Dict({
            "grid": spaces.Box(-1.0, 1.0, (grid_channels, MAX_HEIGHT, MAX_WIDTH), np.float32),
            "scalars": spaces.Box(0.0, 1.0, (scalar_count,), np.float32),
        })
        self.action_space = spaces.Discrete(self.n_actions)

        self._np_rng = np.random.default_rng(seed)
        self.engine: GameEngine | None = None
        self._invalid_streak = 0
        self._idle_streak = 0
        self._prev_dist = 0

    # ---- encodage (identique au contrat 15 canaux d'AlgoGamesEnv) ---------
    def _build_grid(self):
        return build_grid(self.engine, self.hero, self.elevation, self.engine_config, self.grid_channels)

    def _build_scalars(self):
        return build_scalars(self.engine, self.hero, self.scalar_count)

    def _obs(self):
        return {"grid": self._build_grid(), "scalars": self._build_scalars()}

    def _info(self):
        e = self.engine
        return {
            "is_on_engine": e.on_engine[self.hero] is not None,
            "hero_pos": e.pos[self.hero],
            "hero_stamina": e.stamina[self.hero],
            "hero_battery": e.battery[self.hero],
            "stones_collected": e.stones_collected,
            "chests_hidden": e.chests_hidden,
            "chests_destroyed": e.chests_destroyed,
        }

    def _nearest_objective_dist(self):
        e = self.engine
        hx, hy = e.pos[self.hero]
        targets = list(e.stones) + [(c[0], c[1]) for c in e.chests]
        if not targets:
            return 0
        return min(abs(tx - hx) + abs(ty - hy) for tx, ty in targets)

    # ---- gym API ------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._np_rng = np.random.default_rng(seed)
        self.engine = GameEngine(
            self.ascii_rows, self.elevation, self.max_time,
            engine_config=self.engine_config,
            seed=int(self._np_rng.integers(0, 2**31 - 1)),
        )
        self._invalid_streak = 0
        self._idle_streak = 0
        self._prev_dist = self._nearest_objective_dist()
        return self._obs(), self._info()

    def step(self, action):
        e = self.engine
        a_name = self.action_names[int(action)]
        other_action = e.scripted_action(self.other)
        # GameEngine invalide deja nativement toute action hors du contrat
        # action_mask() (OFF_ENGINE_ACTIONS vs HACK_*), pas besoin de le
        # revalider ici.
        ev = e.step({self.hero: a_name, self.other: other_action})
        reward = self._reward(ev[self.hero], ev[self.other])

        resource_low = e.resource_low_ticks[self.hero] >= _RESOURCE_LOW_TICKS_LIMIT
        cleared = not e.stones and not e.chests
        timeout = e.tick >= e.max_time
        terminated = bool(resource_low or cleared)
        truncated = bool(timeout and not terminated)

        info = self._info()
        if resource_low:
            info["resources_exhausted"] = True
            reward += self.reward_config["resource_exhausted"]
        if terminated or truncated:
            if truncated:
                reward += self.reward_config.get("timeout", 0.0)
            reward += e.official_score() * _SCORE_BONUS_SCALE

        return self._obs(), float(reward), terminated, truncated, info

    def _reward(self, my, other_ev):
        rc = self.reward_config
        if not my["valid"]:
            r = rc["invalid_action"]
            self._invalid_streak += 1
            self._idle_streak = 0  # une tentative (meme ratee) n'est pas de la passivite
            if self._invalid_streak >= _STREAK_LIMIT:
                r += rc["prolonged_block"]
            return r
        self._invalid_streak = 0
        r = 0.0
        kind = my["kind"]

        # Suivi de la serie de WAIT consecutifs (kind=="wait" = attente HORS
        # engine, choix delibere de ne rien faire ; distinct de "unhack"/
        # "ride_wait" qui sont des ticks forces par la mecanique et ne
        # doivent pas etre penalises comme de la passivite).
        if kind == "wait":
            self._idle_streak += 1
        else:
            self._idle_streak = 0

        if kind == "wait":
            if self.engine.stamina[self.hero] >= 100.0:
                r += rc["idle_action"]
            if self._idle_streak > _IDLE_STREAK_LIMIT:
                r += rc["idle_streak_penalty"]
        elif kind == "hack":
            r += rc["useful_hack"]
        elif kind == "hack_blocked_rotate":
            r += rc["invalid_action"]
        elif kind in ("hack_move_queued", "hack_cw", "hack_ccw"):
            r += rc["strategic_action"]
        elif kind == "fill":
            r += rc["useful_fill"]
        elif kind == "cut":
            r += rc["useful_cut"]
        elif kind == "chest_pushed":
            r += rc["useful_push"]
        elif kind == "chest_hidden":
            r += rc["chest_hidden"]
        elif kind in ("chest_hole", "chest_cliff"):
            r += rc["voluntary_chest_loss"]
        if my.get("stone"):
            r += rc["stone_collected"]
        if other_ev["kind"] in ("chest_hole", "chest_cliff"):
            r += rc["chest_destroyed"]
        dist = self._nearest_objective_dist()
        r += rc["progress_to_objective"] * max(-1, min(1, self._prev_dist - dist))
        self._prev_dist = dist
        return r


def make_env(**kwargs) -> AlgoEnv:
    return AlgoEnv(**kwargs)
