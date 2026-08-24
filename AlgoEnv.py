"""AlgoEnv.py — Environnement gymnasium reel pour AlgoGames 2 (GDD + Twist).

Encodage 15 canaux IDENTIQUE au contrat documente dans AlgoTrain.AlgoGamesEnv
(meme mapping de tuiles, memes formules de normalisation d'elevation), mais
anime par GameEngine.py : vraies regles (F/M/X/G, hop/climb, push+Twist,
hacking, autopilote machines, look-ahead collisions) au lieu du squelette.

Charge dynamiquement par AlgoTrain.main() via find_factory() -> make_env().
"""
from __future__ import annotations
from collections import deque
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
# Fenetre glissante pour la detection d'oscillation : si le hero revient sur
# une position qu'il a occupee dans les _POSITION_HISTORY_LEN derniers deplacements,
# on declenche la penalite revisited_position. K=6 capture les cycles 2-6,
# laisse assez de marge pour re-emprunter un chemin legitime apres 6+ pas.
_POSITION_HISTORY_LEN = 6
# Le score officiel est injecte a la fin sous forme de difference avec le
# score initial. Cela evite un bonus constant important, independant des
# actions, et donne exactement la variation de performance officielle.
_TERMINAL_SCORE_DELTA_SCALE = 1.0
_REWARD_DEFAULTS = {
    # Evenements officiels.
    "stone_collected": 25.0,
    "chest_hidden": 150.0,
    # Shaping potentiel vers les objectifs.
    "progress_to_stone": 0.50,
    "regress_from_stone": -0.50,
    "progress_to_chest": 0.60,
    "regress_from_chest": -0.60,
    "progress_chest_to_bush": 1.00,
    "regress_chest_from_bush": -0.75,
    # Machines.
    "useful_hack": 0.15,
    "useful_fill": 0.75,
    "useful_cut": 0.75,
    # WAIT contextuel — assoupli vs la 1re fix trop agressive.
    # RATIONNEL : la 1re fix poussait wait_productive_action_available a -1.50
    # pour casser le farm WAIT. Mais combinée au shaping symétrique (aller-retour
    # ~0), cela rendait l'oscillation (-0.05/pas) bien meilleure marche que WAIT
    # (-1.55/pas) -> l'agent oscillait au lieu d'attendre. La penalite de
    # revisite (revisited_position ci-dessous) attaque desormais l'oscillation
    # directement, ce qui permet de ramener WAIT a un niveau modere qui pousse
    # a l'exploration sans dominer les autres signaux. Ordre partiel cible :
    #   avancer (+1.05) > WAIT (-0.75) > osciller (-1.50 a -2.05).
    "step_time_cost": -0.05,
    "wait_recovery_per_stamina": 0.10,
    "wait_forced": 0.0,
    "wait_useful_machine": 0.10,
    "wait_unhack": -0.03,
    "wait_no_productive_action": -0.05,
    "wait_productive_action_available": -0.75,   # etait -1.50 -> assoupli
    "wait_full_resources": -1.00,                # etait -2.00 -> assoupli
    "repeated_wait_2": -0.50,                    # etait -0.75 -> assoupli
    "repeated_wait_3_plus": -1.25,               # etait -2.00 -> assoupli
    # Anti-oscillation directe : penalite lourde des qu'on revient sur une
    # position occupee dans les _POSITION_HISTORY_LEN derniers deplacements.
    # Contrairement a un shaping asymetrique (qui cree du farming), celle-ci
    # est asymetrique par construction (ne s'applique qu'au retour, jamais a
    # l'aller) et donc compatible avec le telescoping du _potential().
    "revisited_position": -2.00,
    # Invalidite et gaspillage.
    "invalid_action": -0.25,
    "repeated_invalid_action": -0.50,
    "push_without_chest": -0.35,
    "blocked_push": -0.25,
    "hack_without_machine": -0.35,
    "hack_action_without_riding": -0.35,
    "useless_hack_rotation": -0.10,
    "blocked_hack_move": -0.20,
    "wasted_battery": -0.25,
    # Perte d'un objectif.
    "chest_destroyed": -25.0,
    "voluntary_chest_loss": -50.0,
    # Fin de partie.
    "resource_exhausted": -5.0,
    "timeout": 0.0,
    # Additional shaping / penalties (expanded strategy)
    "implicit_wait": -0.20,
    "blocked_move": -0.15,
    "insufficient_stamina": -0.10,
    "wrong_machine_type": -0.25,
    "hack_when_machine_needed": 0.50,
    "hack_when_machine_not_needed": -0.25,
    "rotation_toward_target": 0.20,
    "rotation_away_from_target": -0.15,
    "hack_move_progress": 0.35,
    "hack_move_regress": -0.20,
    "fill_useful_hole": 2.00,
    "fill_irrelevant_hole": -0.10,
    "cut_useful_tree": 2.00,
    "cut_irrelevant_tree": -0.10,
    "resource_score_delta_scale": 0.25,
    "machine_stone_recovered": 25.0,
    "machine_stone_stolen_by_other": 0.0,
    "objectives_cleared": 50.0,                  # etait 20.0 -> bonus terminal renforce
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
        # Fenetre glissante des dernieres positions occupees par le hero,
        # pour detecter les oscillations (cycles 2-6) et appliquer la
        # penalite revisited_position. Voir _POSITION_HISTORY_LEN.
        self._position_history: deque[tuple[int, int]] = deque(maxlen=_POSITION_HISTORY_LEN)

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
            # Masque structurel (autorise les erreurs physiques qui seront
            # converties en WAIT par le moteur) : preferer pour l'apprentissage.
            "action_mask": e.structural_action_mask(self.hero, self.action_names),
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

    def _has_productive_action(self) -> bool:
        """Indique si le hero a une action utile autre que WAIT/HACK_CW/HACK_CCW.

        Detection elargie par rapport au masque par defaut : on regarde aussi
        PUSH_* (pousser un coffre vers une buisson) et HACK (monter sur une
        machine cible) hors engine, ainsi que HACK_MOVE / HACK_FILL sur engine.
        Cela evite que la penalite `wait_productive_action_available` ne tombe a
        -0.10 (wait_no_productive_action) quand une action utile existe en
        realite : cf. diagnostic Cause 4.
        """
        e = self.engine
        mask = e.legal_action_mask(self.hero, self.action_names)
        riding = e.on_engine[self.hero] is not None
        for index, name in enumerate(self.action_names):
            if not mask[index] or name == "WAIT":
                continue
            if name in ("HACK_CW", "HACK_CCW"):
                continue
            # Sur engine : HACK_MOVE et HACK_FILL sont productifs.
            if riding and name in ("HACK_MOVE", "HACK_FILL"):
                return True
            # Hors engine : MOVE, PUSH, HACK sont productifs.
            if not riding and name in (
                "UP", "DOWN", "LEFT", "RIGHT",
                "PUSH_UP", "PUSH_DOWN", "PUSH_LEFT", "PUSH_RIGHT",
                "HACK",
            ):
                return True
        return False

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
        # Track local resource potential for terminal credit assignment
        self._initial_local_resource_score = self._hero_resource_score()
        # potential-based shaping
        self._prev_potential = self._potential()
        self._ep_len = 0
        self._ep_wait = 0
        self._ep_invalid = 0
        # Initialise l'historique des positions avec la position de depart.
        # Le deque evolue au fil des step() : on n'ajoute QUE les positions
        # differentes de la precedente (WAIT n'enrichit pas l'historique,
        # sinon le farm WAIT declencherait faussement la penalite).
        self._position_history.clear()
        self._position_history.append(tuple(self.engine.pos[self.hero]))
        return self._obs(), self._info()

    def step(self, action):
        e = self.engine
        action_index = int(action)
        a_name = self.action_names[action_index]
        precise_mask = e.legal_action_mask(self.hero, self.action_names)
        selected_is_legal = bool(precise_mask[action_index])
        # Position avant step : sert a detecter un deplacement effectif
        # (different de WAIT/implicit_wait/invalid) pour mettre a jour
        # l'historique des positions et appliquer la penalite de revisite.
        before_pos = tuple(e.pos[self.hero])
        before_chests = {
            (int(cx), int(cy))
            for cx, cy in e.chests
        }
        before_hidden = e.chests_hidden
        before_destroyed = e.chests_destroyed
        before_battery = e.battery[self.hero]
        before_stamina = e.stamina[self.hero]
        productive_action_available = self._has_productive_action()
        other_action = e.scripted_action(self.other)
        ev = e.step({self.hero: a_name, self.other: other_action})
        my = ev[self.hero]
        other_ev = ev[self.other]
        after_pos = tuple(e.pos[self.hero])
        moved = (after_pos != before_pos)
        # Penalite anti-oscillation : declenchee uniquement si le hero s'est
        # DEPLACE sur une case qu'il a deja occupee dans les K derniers pas.
        # WAIT/invalid/implicit_wait ne declenchent pas la penalite (sinon
        # le farm WAIT cumulerait revisite + wait_penalty, double peine).
        revisited = moved and (after_pos in self._position_history)
        reward = self._reward(
            my=my,
            other_ev=other_ev,
            requested_action=a_name,
            selected_is_legal=selected_is_legal,
            before_chests=before_chests,
            before_hidden=before_hidden,
            before_destroyed=before_destroyed,
            before_battery=before_battery,
            before_stamina=before_stamina,
            productive_action_available=productive_action_available,
            revisited=revisited,
        )
        # Mise a jour de l'historique : on ne stocke QUE les positions
        # effectivement atteintes par un deplacement, pas les WAIT (sinon
        # on polluerait le detecteur avec des positions identiques).
        if moved:
            self._position_history.append(after_pos)
        # potential shaping (telescoping) to encourage genuine progress.
        # Forme canonique F(s,a,s') = gamma*Phi(s') - Phi(s) (Ng, Harada &
        # Russell 1999) : sans le gamma, un aller-retour ne s'annule plus
        # exactement des que gamma != 1, ce qui rouvrait (en plus petit) le
        # meme farm que le bloc distance_reward retire ci-dessous.
        reward_gamma = float(self.engine_config.get("reward_gamma", 1.0))
        current_potential = self._potential()
        reward += reward_gamma * current_potential - self._prev_potential
        self._prev_potential = current_potential
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
        # By default keep running until timeout; early stop on cleared optional
        early_stop_on_cleared = self.engine_config.get("early_stop_on_cleared", False)
        terminated = bool(resource_low and self.engine_config.get("resource_exhaustion_first", True))
        objectives_just_cleared = early_stop_on_cleared and cleared and not terminated
        if objectives_just_cleared:
            terminated = True
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
            if objectives_just_cleared:
                # Bonus declare depuis le debut (_REWARD_DEFAULTS /
                # DEFAULT_REWARD_CONFIG) mais jamais applique auparavant.
                reward += self.reward_config.get("objectives_cleared", 0.0)
        elif truncated:
            info["termination_reason"] = "timeout"
            reward += self.reward_config.get("timeout", 0.0)
        if terminated or truncated:
            # Assign local resource delta to avoid credit assignment to the other hero
            local_resource_delta = self._hero_resource_score() - self._initial_local_resource_score
            reward += local_resource_delta
            final_score = float(e.official_score())
            score_delta = final_score - self._initial_official_score
            info["official_score"] = final_score
            info["official_score_delta"] = score_delta
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
        before_stamina: float,
        productive_action_available: bool,
        revisited: bool = False,
    ):
        rc = self.reward_config
        e = self.engine
        reward = rc.get("step_time_cost", -0.05)
        # Penalite anti-oscillation : s'applique AVANT la branche invalid/wait
        # car elle reflete un deplacement reel vers une position recente.
        # Sur une action invalide, le moteur ne bouge pas le hero (implicit_wait)
        # donc revisited est necessairement False -> pas de double peine.
        if revisited:
            reward += rc["revisited_position"]
        if not selected_is_legal or not my["valid"]:
            self._invalid_streak += 1
            reward += self._invalid_penalty(requested_action, my["kind"])
            if self._invalid_streak >= _STREAK_LIMIT:
                reward += rc["repeated_invalid_action"]
            self._refresh_objective_distances()
            return reward
        self._invalid_streak = 0
        kind = my["kind"]
        if kind == "wait":
            recovered = max(0.0, e.stamina[self.hero] - before_stamina)
            if recovered > 0.0:
                reward += rc["wait_recovery_per_stamina"] * recovered
                self._idle_streak = 0
            else:
                self._idle_streak += 1
                resources_full = (
                    e.stamina[self.hero] >= 99.999
                    and e.battery[self.hero] >= 99.999
                )
                if resources_full:
                    reward += rc["wait_full_resources"]
                elif productive_action_available:
                    reward += rc["wait_productive_action_available"]
                else:
                    reward += rc["wait_no_productive_action"]
                if self._idle_streak == 2:
                    reward += rc["repeated_wait_2"]
                elif self._idle_streak >= 3:
                    reward += rc["repeated_wait_3_plus"]
        elif kind == "ride_wait":
            reward += rc["wait_forced"]
            self._idle_streak = 0
        elif kind == "cut":
            reward += rc["useful_cut"]
            self._idle_streak = 0
        elif kind == "unhack":
            reward += rc["wait_unhack"]
            self._idle_streak = 0
        elif kind == "hack":
            reward += 0.0
            self._idle_streak = 0
        elif kind == "hack_blocked_rotate":
            reward += rc["blocked_hack_move"]
            self._idle_streak = 0
        elif kind in ("hack_cw", "hack_ccw"):
            reward += rc["useless_hack_rotation"]
            self._idle_streak = 0
        elif kind == "fill":
            reward += rc["useful_fill"]
            self._idle_streak = 0
        elif kind == "chest_hidden":
            reward += rc["chest_hidden"]
            self._idle_streak = 0
        elif kind in ("chest_hole", "chest_cliff"):
            reward += rc["voluntary_chest_loss"]
            self._idle_streak = 0
        else:
            self._idle_streak = 0
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
            "unhack",
        }
        if battery_spent > 0.0 and kind not in useful_battery_kinds:
            reward += rc["wasted_battery"] * battery_spent
        # --- Shaping distance pas-a-pas (coefficients SYMETRIQUES) ---
        # Un aller-retour donne un gain net de zero -> pas d'oscillation.
        # Le bloc precedent etait commentaire (cf. diagnostic Cause 3) ce qui
        # privait l'agent de tout feedback immediat sur la qualite de ses
        # mouvements. On reintroduit le shaping avec progress = |regress| pour
        # neutraliser le farming oscillant, et on l'exclut sur les kinds non
        # volontaires (wait, implicit_wait, invalid, hack_blocked_rotate,
        # unhack, ride_wait, chest_*) pour ne pas bruiter le signal avec des
        # evenements qui ne changent pas reellement la distance.
        new_stone_dist = self._nearest_stone_dist()
        new_chest_dist = self._nearest_chest_dist()
        if kind not in (
            "chest_pushed", "chest_hole", "chest_cliff",
            "wait", "ride_wait", "unhack", "hack_blocked_rotate",
            "implicit_wait", "invalid",
        ):
            reward += self._distance_reward(
                self._prev_stone_dist, new_stone_dist,
                rc["progress_to_stone"], rc["regress_from_stone"],
            )
            reward += self._distance_reward(
                self._prev_chest_dist, new_chest_dist,
                rc["progress_to_chest"], rc["regress_from_chest"],
            )
        self._prev_stone_dist = new_stone_dist
        self._prev_chest_dist = new_chest_dist
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

    def _hero_resource_score(self) -> float:
        e = self.engine
        return 0.25 * (e.stamina[self.hero] + e.battery[self.hero]) + 0.25 * max(0, e.max_time - e.tick)

    def _potential(self) -> float:
        """Potentiel F(s) pour shaping telescoping F(s,a,s') = gamma*Phi(s') - Phi(s).

        Coefficients x3 par rapport au reglage original (-0.20 -> -0.60 pour
        les pierres, -0.25 -> -0.75 pour les coffres) afin que le gain net
        par pas vers un objectif (0.55-0.70 apres soustraction de step_time_cost)
        surpasse largement la penalite d'action invalide (-0.25). Sans ce
        renforcement, l'agent apprend que WAIT est plus sur que n'importe quel
        mouvement (cf. diagnostic Cause 1).
        """
        e = self.engine
        stone_d = self._nearest_stone_dist()
        chest_d = self._nearest_chest_dist()
        stone_term = 0.0 if stone_d is None else -0.60 * stone_d   # x3
        chest_term = 0.0 if chest_d is None else -0.75 * chest_d   # x3
        resource_term = 0.005 * (e.stamina[self.hero] + e.battery[self.hero])
        return stone_term + chest_term + resource_term

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