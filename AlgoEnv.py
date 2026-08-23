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

_STREAK_LIMIT = 3
_RESOURCE_LOW_TICKS_LIMIT = 10
# Le score officiel est injecte a la fin sous forme de difference avec le
# score initial. Cela evite un bonus constant important, independant des
# actions, et donne exactement la variation de performance officielle.
_TERMINAL_SCORE_DELTA_SCALE = 1.0
_REWARD_DEFAULTS = {
    # Evenements du score officiel.
    "stone_collected": 25.0,
    "chest_hidden": 150.0,
    # Shaping potentiel vers les objectifs qui produisent C et P.
    "progress_to_stone": 0.30,
    "regress_from_stone": -0.30,
    "progress_to_chest": 0.40,
    "regress_from_chest": -0.40,
    "progress_chest_to_bush": 0.75,
    "regress_chest_from_bush": -0.75,
    # Actions machine utiles uniquement si elles ouvrent ou suivent un chemin
    # vers une pierre, un coffre ou une cache.
    "useful_hack": 0.20,
    "useful_fill": 0.50,
    "useful_cut": 0.50,
    # Invalidite et gaspillage.
    "invalid_action": -0.25,
    "repeated_invalid_action": -0.50,
    "wait_with_full_resources": -0.20,
    "wait_without_recovery_need": -0.05,
    "push_without_chest": -0.35,
    "blocked_push": -0.25,
    "hack_without_machine": -0.35,
    "hack_action_without_riding": -0.35,
    "useless_hack_rotation": -0.10,
    "blocked_hack_move": -0.20,
    "wasted_battery": -0.30,
    # Perte d'un objectif.
    "chest_destroyed": -25.0,
    "voluntary_chest_loss": -50.0,
    # Fin de partie.
    "resource_exhausted": -5.0,
    "timeout": 0.0,
}


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
        # Carte de base non-augmentee, conservee pour re-tirer l'augmentation
        # a CHAQUE reset() (cf. diagnostic Cause 3 : figer l'augmentation une
        # seule fois a la creation de l'env biaise tout l'entrainement vers
        # une seule orientation, potentiellement differente de celle utilisee
        # a l'evaluation Optuna qui, elle, force toujours "identity").
        self._base_rows, self._base_elev = base_rows, base_elev
        self.augmentation_mode = augmentation
        self.augmentations = augmentations
        self._aug_cycle = -1
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
        self._prev_stone_dist: int | None = None
        self._prev_chest_dist: int | None = None
        self._initial_official_score = 0.0
        self._ep_len = 0
        self._ep_wait = 0
        self._ep_invalid = 0

    # ---- encodage (identique au contrat 15 canaux d'AlgoGamesEnv) ---------
    def _build_grid(self):
        return build_grid(self.engine, self.hero, self.elevation, self.engine_config, self.grid_channels)

    def _build_scalars(self):
        return build_scalars(self.engine, self.hero, self.scalar_count)

    def _obs(self):
        return {"grid": self._build_grid(), "scalars": self._build_scalars()}

    def _info(self):
        e = self.engine
        ep_len = max(1, self._ep_len)
        return {
            "is_on_engine": e.on_engine[self.hero] is not None,
            "hero_pos": e.pos[self.hero],
            "hero_stamina": e.stamina[self.hero],
            "hero_battery": e.battery[self.hero],
            "stones_collected": e.stones_collected,
            "chests_hidden": e.chests_hidden,
            "chests_destroyed": e.chests_destroyed,
            # Masque precis (cf. GameEngine.legal_action_mask) : lu en priorite
            # par ActionMasker._mask_from_info(), remplace le fallback grossier
            # base uniquement sur is_on_engine.
            "action_mask": e.legal_action_mask(self.hero, self.action_names),
            "wait_ratio": self._ep_wait / ep_len,
            "invalid_ratio": self._ep_invalid / ep_len,
        }

    def _pick_augmentation(self) -> str:
        mode = self.augmentation_mode
        if mode == "random":
            return str(self._np_rng.choice(self.augmentations))
        if mode == "all":
            self._aug_cycle = (self._aug_cycle + 1) % len(self.augmentations)
            return self.augmentations[self._aug_cycle]
        return mode if mode in self.augmentations else "identity"

    def _nearest_distance(self, targets):
        e = self.engine
        if not targets:
            return None
        hx, hy = e.pos[self.hero]
        return min(abs(tx - hx) + abs(ty - hy) for tx, ty in targets)
    def _nearest_stone_dist(self):
        return self._nearest_distance(self.engine.stones)
    def _nearest_chest_dist(self):
        return self._nearest_distance(
            {(int(cx), int(cy)) for cx, cy in self.engine.chests}
        )
    def _nearest_bush_distance(self, x: int, y: int) -> int | None:
        bushes = {
            (bx, by)
            for by, row in enumerate(self.engine.terrain)
            for bx, tile in enumerate(row)
            if tile == "@"
        }
        if not bushes:
            return None
        return min(abs(bx - x) + abs(by - y) for bx, by in bushes)
    @staticmethod
    def _distance_reward(
        previous: int | None,
        current: int | None,
        progress_reward: float,
        regress_penalty: float,
    ) -> float:
        if previous is None or current is None:
            return 0.0
        if current < previous:
            return progress_reward
        if current > previous:
            return regress_penalty
        return 0.0
    def _refresh_objective_distances(self) -> None:
        self._prev_stone_dist = self._nearest_stone_dist()
        self._prev_chest_dist = self._nearest_chest_dist()

    # ---- gym API ------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._np_rng = np.random.default_rng(seed)
        mode = self._pick_augmentation()
        self.ascii_rows, self.elevation = _augment(self._base_rows, self._base_elev, mode)
        self.height, self.width = len(self.ascii_rows), len(self.ascii_rows[0])
        self.engine = GameEngine(
            self.ascii_rows, self.elevation, self.max_time,
            engine_config=self.engine_config,
            seed=int(self._np_rng.integers(0, 2**31 - 1)),
        )
        self._invalid_streak = 0
        self._idle_streak = 0
        self._refresh_objective_distances()
        self._initial_official_score = float(self.engine.official_score())
        self._ep_len = 0
        self._ep_wait = 0
        self._ep_invalid = 0
        return self._obs(), self._info()

    def step(self, action):
        e = self.engine
        action_index = int(action)
        a_name = self.action_names[action_index]
        precise_mask = e.legal_action_mask(self.hero, self.action_names)
        selected_is_legal = bool(precise_mask[action_index])
        before_chests = {
            (int(cx), int(cy))
            for cx, cy in e.chests
        }
        before_hidden = e.chests_hidden
        before_destroyed = e.chests_destroyed
        before_battery = e.battery[self.hero]
        other_action = e.scripted_action(self.other)
        ev = e.step({self.hero: a_name, self.other: other_action})
        my = ev[self.hero]
        other_ev = ev[self.other]
        reward = self._reward(
            my=my,
            other_ev=other_ev,
            requested_action=a_name,
            selected_is_legal=selected_is_legal,
            before_chests=before_chests,
            before_hidden=before_hidden,
            before_destroyed=before_destroyed,
            before_battery=before_battery,
        )
        self._ep_len += 1
        if not my["valid"]:
            self._ep_invalid += 1
        elif my["kind"] == "wait":
            self._ep_wait += 1
        resource_low = (
            e.resource_low_ticks[self.hero] >= _RESOURCE_LOW_TICKS_LIMIT
        )
        cleared = not e.stones and not e.chests
        timeout = e.tick >= e.max_time
        terminated = bool(resource_low or cleared)
        truncated = bool(timeout and not terminated)
        info = self._info()
        info["action_name"] = a_name
        info["action_result"] = my["kind"]
        if resource_low:
            info["resources_exhausted"] = True
            info["termination_reason"] = "resources_exhausted"
            reward += self.reward_config["resource_exhausted"]
        elif cleared:
            info["termination_reason"] = "objectives_cleared"
        elif truncated:
            info["termination_reason"] = "timeout"
            reward += self.reward_config.get("timeout", 0.0)
        if terminated or truncated:
            final_score = float(e.official_score())
            score_delta = final_score - self._initial_official_score
            # Les pierres et coffres ont deja produit leurs rewards exacts durant
            # l'episode. Pour eviter le double comptage, le bonus terminal ne
            # conserve que l'effet ressources + temps non deja redistribue.
            direct_objective_score = (
                e.stones_collected * 25.0
                + e.chests_hidden * 150.0
            )
            residual_score_delta = score_delta - direct_objective_score
            reward += residual_score_delta * _TERMINAL_SCORE_DELTA_SCALE
            info["official_score"] = final_score
            info["official_score_delta"] = score_delta
            info["terminal_residual_score_delta"] = residual_score_delta
        return self._obs(), float(reward), terminated, truncated, info

    def _reward(
        self,
        my,
        other_ev,
        requested_action: str,
        selected_is_legal: bool,
        before_chests: set[tuple[int, int]],
        before_hidden: int,
        before_destroyed: int,
        before_battery: float,
    ):
        rc = self.reward_config
        e = self.engine
        if not selected_is_legal or not my["valid"]:
            self._invalid_streak += 1
            reward = self._invalid_penalty(requested_action, my["kind"])
            if self._invalid_streak >= _STREAK_LIMIT:
                reward += rc["repeated_invalid_action"]
            self._refresh_objective_distances()
            return reward
        self._invalid_streak = 0
        reward = 0.0
        kind = my["kind"]
        if kind == "wait":
            if e.stamina[self.hero] <= 0.0 or e.battery[self.hero] <= 0.0:
                reward += rc["wait_with_full_resources"]
        elif kind == "hack":
            reward += rc["useful_hack"]
        elif kind == "hack_blocked_rotate":
            reward += rc["blocked_hack_move"]
        elif kind in ("hack_cw", "hack_ccw"):
            reward += rc["useless_hack_rotation"]
        elif kind == "fill":
            reward += rc["useful_fill"]
        elif kind == "cut":
            reward += rc["useful_cut"]
        elif kind == "chest_hidden":
            reward += rc["chest_hidden"]
        elif kind in ("chest_hole", "chest_cliff"):
            reward += rc["voluntary_chest_loss"]
        if my.get("stone"):
            reward += rc["stone_collected"]
        destroyed_delta = e.chests_destroyed - before_destroyed
        if destroyed_delta > 0 and kind not in ("chest_hole", "chest_cliff"):
            reward += rc["chest_destroyed"] * destroyed_delta
        battery_spent = max(0.0, before_battery - e.battery[self.hero])
        useful_battery_kinds = {
            "hack",
            "hack_move_queued",
            "ride_wait",
            "fill",
            "cut",
        }
        if battery_spent > 0.0 and kind not in useful_battery_kinds:
            reward += rc["wasted_battery"] * battery_spent
        current_stone_dist = self._nearest_stone_dist()
        current_chest_dist = self._nearest_chest_dist()
        reward += self._distance_reward(
            self._prev_stone_dist,
            current_stone_dist,
            rc["progress_to_stone"],
            rc["regress_from_stone"],
        )
        reward += self._distance_reward(
            self._prev_chest_dist,
            current_chest_dist,
            rc["progress_to_chest"],
            rc["regress_from_chest"],
        )
        self._prev_stone_dist = current_stone_dist
        self._prev_chest_dist = current_chest_dist
        after_chests = {
            (int(cx), int(cy))
            for cx, cy in e.chests
        }
        if kind == "chest_pushed":
            removed = before_chests - after_chests
            added = after_chests - before_chests
            if len(removed) == 1 and len(added) == 1:
                old_pos = next(iter(removed))
                new_pos = next(iter(added))
                old_dist = self._nearest_bush_distance(*old_pos)
                new_dist = self._nearest_bush_distance(*new_pos)
                if (
                    old_dist is not None
                    and new_dist is not None
                    and new_dist < old_dist
                ):
                    reward += rc["progress_chest_to_bush"]
                elif (
                    old_dist is not None
                    and new_dist is not None
                    and new_dist > old_dist
                ):
                    reward += rc["regress_chest_from_bush"]
        return reward

    def _invalid_penalty(self, requested_action: str, result_kind: str) -> float:
        rc = self.reward_config
        if requested_action.startswith("PUSH_"):
            dx, dy = {
                "PUSH_UP": (0, -1),
                "PUSH_DOWN": (0, 1),
                "PUSH_LEFT": (-1, 0),
                "PUSH_RIGHT": (1, 0),
            }[requested_action]
            hx, hy = self.engine.pos[self.hero]
            if self.engine.chest_at(hx + dx, hy + dy) is None:
                return rc["push_without_chest"]
            return rc["blocked_push"]
        if requested_action == "HACK":
            return rc["hack_without_machine"]
        if requested_action.startswith("HACK_"):
            if self.engine.on_engine[self.hero] is None:
                return rc["hack_action_without_riding"]
            if requested_action == "HACK_MOVE":
                return rc["blocked_hack_move"]
            return rc["invalid_action"]
        return rc["invalid_action"]


def make_env(**kwargs) -> AlgoEnv:
    return AlgoEnv(**kwargs)
