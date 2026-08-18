#!/usr/bin/env python3
"""
AlgoTrain.py - Entrainement RecurrentPPO pour AlgoGames 2.

On entraine DEUX politiques distinctes (un ONNX par hero) :

    - Ikotofosa (F) : Discrete(14) actions
          4 MOVE  (UP, DOWN, LEFT, RIGHT)
          1 WAIT
          4 PUSH  (PUSH_UP, PUSH_DOWN, PUSH_LEFT, PUSH_RIGHT)
          1 HACK
          4 HACK_* (HACK_MOVE, HACK_FILL, HACK_CW, HACK_CCW)
          = 14

    - Imahaki  (M) : Discrete(13) actions
          4 MOVE
          1 WAIT
          4 PUSH
          1 HACK
          3 HACK_* (HACK_MOVE, HACK_CW, HACK_CCW)
          = 13
          (HACK_CUT est automatique sur les grapplers, donc non emis.)

Observation conforme au preprocesseur C# cible (Twist + look-ahead machines) :
    grid    : float32[15, 30, 30]
              11 canaux one-hot tuiles
            + 1 canal elevation absolue (normalise [-1,1])
            + 1 canal elevation relative (level - mean, clipped [-1,1])
            + 1 canal "next_X" : positions futures des excavateurs (T+lookahead)
            + 1 canal "next_G" : positions futures des grapplers   (T+lookahead)
    scalars : float32[6]
              stamina, batterie, temps, position X/Y, isOnEngine

REGLE IMPORTANTE (look-ahead conditionnel) :
    Les canaux 13 (next_X) et 14 (next_G) ne sont remplis QUE si la case
    devant le hero (hero_pos + facing_direction) contient un coffre '*'.
    Sinon, ces deux canaux restent a zero. Cela evite de polluer l'observation
    avec du bruit de simulation quand le hero n'est pas en situation de push.
    Utiliser la fonction `should_run_lookahead(ascii_map, hero_pos, facing)`
    pour decider d'appeler ou non le simulateur de machines.

Le script peut etre appele deux fois (une fois par hero) :

    python AlgoTrain.py --hero F --map map.txt --elevation elevation.txt --output ModelStates/ikotofosa
    python AlgoTrain.py --hero M --map map.txt --elevation elevation.txt --output ModelStates/imahaki

Puis Exports.py produira ikotofosa.onnx (14 actions) et imahaki.onnx (13 actions),
chacun attend un tenseur d'entree map de forme [1, 15, 30, 30].

L'environnement concret doit etre expose par AlgoEnv.py sous l'une des formes :
    - make_env(... hero="F"|"M" ...)
    - AlgoEnv(... hero="F"|"M" ...)
    - AlgoGamesEnv(... hero="F"|"M" ...)
"""
from __future__ import annotations

import argparse
import importlib.util
import inspect
import os
import random
from pathlib import Path
from typing import Any, Callable

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from sb3_contrib import RecurrentPPO
from sb3_contrib.ppo_recurrent.policies import MultiInputLstmPolicy
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

MAX_HEIGHT = 30
MAX_WIDTH = 30
MAX_SOURCE_HEIGHT = 15
MAX_SOURCE_WIDTH = 20

# Twist : 15 canaux d'observation = 11 one-hot tuiles + 1 abs elevation + 1 rel
# elevation + 1 next_X (look-ahead excavateur) + 1 next_G (look-ahead grappler).
N_GRID_CHANNELS = 15
N_TILE_CHANNELS = 11
N_SCALARS = 6
LOOKAHEAD_TICKS = 2  # doir matcher TwistConfig.LookaheadTicks côté C#

ASCII_TILES = frozenset(".#to*+@FMXG")
AUGMENTATIONS = (
    "identity",
    "transpose",
    "rotate90",
    "rotate180",
    "rotate270",
    "mirror_horizontal",
    "mirror_vertical",
)

# ---------------------------------------------------------------------------
# Catalogue d'actions par hero
# ---------------------------------------------------------------------------
# L'ordre des actions est CONTRACTUEL : il doit matcher celui attendu par le
# moteur C# lors de l'inference ONNX. Si vous modifiez l'ordre ici, mettez a
# jour le switch cote C# en consequence.
IKOTOFOSA_ACTIONS: list[str] = [
    "UP", "DOWN", "LEFT", "RIGHT",
    "WAIT",
    "PUSH_UP", "PUSH_DOWN", "PUSH_LEFT", "PUSH_RIGHT",
    "HACK",
    "HACK_MOVE", "HACK_FILL", "HACK_CW", "HACK_CCW",
]
IMAHAKI_ACTIONS: list[str] = [
    "UP", "DOWN", "LEFT", "RIGHT",
    "WAIT",
    "PUSH_UP", "PUSH_DOWN", "PUSH_LEFT", "PUSH_RIGHT",
    "HACK",
    "HACK_MOVE", "HACK_CW", "HACK_CCW",
]

HEROES: dict[str, dict[str, Any]] = {
    "F": {
        "name": "Ikotofosa",
        "actions": IKOTOFOSA_ACTIONS,
        "n_actions": len(IKOTOFOSA_ACTIONS),  # 14
        "ascii": "F",
        "machine_target": "X",  # excavateur -> HACK_FILL
    },
    "M": {
        "name": "Imahaki",
        "actions": IMAHAKI_ACTIONS,
        "n_actions": len(IMAHAKI_ACTIONS),  # 13
        "ascii": "M",
        "machine_target": "G",  # grappler -> HACK_CUT auto, pas d'action explicite
    },
}

# Parametres transmis a AlgoEnv/GameEngine. L'environnement reste l'autorite
# des collisions, actions automatiques, etats hackes et regles du Twist.
DEFAULT_ENGINE_CONFIG = {
    "look_ahead": True,
    "lookahead_ticks": 2,
    # REGLE (twist conditionnel) : les canaux next_X / next_G ne sont remplis
    # QUE si un coffre ('*') se trouve devant le hero. Sinon, les 2 canaux
    # restent a zero. Cela correspond a la regle appliquee cote Godot.
    "lookahead_only_if_chest_ahead": True,
    "avoid_chest_collisions": True,
    "avoid_agent_collisions": True,
    "avoid_competing_pushers": True,
    "fallback_rotation": True,
    "fallback_when_hacked": True,
    "hero_height_block": 2,
    "machine_height_block": 5,
    "uphill_stamina_cost": 3.0,
    "downhill_stamina_cost": 0.0,
    "forbid_chest_uphill": True,
    "destroy_chest_drop": 5,
    "resource_exhaustion_first": True,
}

DEFAULT_REWARD_CONFIG = {
    # Objectifs principaux.
    "stone_collected": 25.0,
    "chest_hidden": 150.0,

    # Etapes strategiques.
    "strategic_action": 0.30,
    "useful_hack": 0.75,
    "useful_fill": 1.00,
    "useful_cut": 1.00,
    "useful_push": 0.50,
    "progress_to_objective": 0.05,

    # Penalites.
    "invalid_action": -0.15,
    "idle_action": -0.02,
    "prolonged_block": -1.00,
    "chest_destroyed": -35.0,
    "voluntary_chest_loss": -75.0,
    "resource_exhausted": -5.0,
    "timeout": 0.0,
}


class GridScalarExtractor(BaseFeaturesExtractor):
    """Modele fourni, conserve sans modification architecturale.

    Le features_dim de sortie (128 par defaut) alimente la tete LSTM de
    RecurrentPPO. La dimension d'entree (13 canaux + 6 scalaires) est
    identique pour les deux heroes, donc l'extracteur est partage.
    """

    def __init__(self, observation_space, features_dim: int = 128):
        super().__init__(observation_space, features_dim)

        self.cnn = nn.Sequential(
            nn.Conv2d(N_GRID_CHANNELS, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

        self.mlp = nn.Sequential(
            nn.Linear(N_SCALARS, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )

        self.combined = nn.Sequential(
            nn.Linear(64 + 32, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations):
        grid_feat = self.cnn(observations["grid"])
        scalar_feat = self.mlp(observations["scalars"])
        return self.combined(torch.cat([grid_feat, scalar_feat], dim=1))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Entraine RecurrentPPO sur AlgoGames 2 (un modele par hero)."
    )
    parser.add_argument(
        "--hero",
        required=True,
        choices=("F", "M"),
        help="Hero a entrainer : F=Ikotofosa (14 actions), M=Imahaki (13 actions).",
    )
    parser.add_argument("--map", required=True, dest="map_path")
    parser.add_argument("--elevation", required=True, dest="elevation_path")
    parser.add_argument(
        "--output",
        required=True,
        help="Chemin de sortie (sans extension). Le .zip est ajoute automatiquement.",
    )
    parser.add_argument(
        "--preprocessor",
        required=False,
        help="(deprecated) Chemin de MapPreprocessor.py. Ignore quand integre.",
    )

    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--eval-freq", type=int, default=25_000)
    parser.add_argument("--checkpoint-freq", type=int, default=50_000)
    parser.add_argument("--resume", help="Modele .zip a reprendre.")
    parser.add_argument(
        "--augmentation",
        choices=("random", "all", *AUGMENTATIONS),
        default="random",
    )
    parser.add_argument("--check-env", action="store_true")
    return parser.parse_args()


def resolve_file(path: str, label: str) -> Path:
    result = Path(path).expanduser().resolve()
    if not result.is_file():
        raise FileNotFoundError(f"{label} introuvable : {result}")
    return result


def read_map_header(path: Path) -> tuple[int, int, int, list[str]]:
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    if not lines:
        raise ValueError("map.txt est vide.")

    header = lines[0].split()
    if len(header) != 3:
        raise ValueError("La premiere ligne de map.txt doit etre : H W temps.")

    try:
        height, width, max_time = map(int, header)
    except ValueError as exc:
        raise ValueError("H, W et temps doivent etre des entiers.") from exc

    if not 1 <= height <= MAX_SOURCE_HEIGHT:
        raise ValueError(f"H doit etre compris entre 1 et {MAX_SOURCE_HEIGHT}.")
    if not 1 <= width <= MAX_SOURCE_WIDTH:
        raise ValueError(f"W doit etre compris entre 1 et {MAX_SOURCE_WIDTH}.")
    if max_time <= 0:
        raise ValueError("La limite de temps doit etre strictement positive.")

    rows = lines[1:1 + height]
    if len(rows) != height or any(len(row) != width for row in rows):
        raise ValueError("Les dimensions ASCII ne correspondent pas a H et W.")

    unknown = sorted({char for row in rows for char in row} - ASCII_TILES)
    if unknown:
        raise ValueError(f"Symboles ASCII inconnus : {unknown}")

    if sum(row.count("F") for row in rows) != 1:
        raise ValueError("La carte doit contenir exactement un F.")
    if sum(row.count("M") for row in rows) != 1:
        raise ValueError("La carte doit contenir exactement un M.")

    return height, width, max_time, rows


def read_elevation(path: Path, height: int, width: int) -> np.ndarray:
    rows: list[list[int]] = []

    for line in path.read_text(encoding="utf-8-sig").splitlines():
        stripped = line.strip()
        if not stripped:
            continue

        parts = stripped.replace(",", " ").split()
        if len(parts) == 1 and len(parts[0]) == width and parts[0].isdigit():
            values = [int(char) for char in parts[0]]
        else:
            try:
                values = [int(value) for value in parts]
            except ValueError as exc:
                raise ValueError(
                    "elevation.txt ne doit contenir que des valeurs 0 a 9."
                ) from exc
        rows.append(values)

    if len(rows) != height or any(len(row) != width for row in rows):
        raise ValueError(
            "Les dimensions d'elevation.txt doivent correspondre a map.txt."
        )

    elevation = np.asarray(rows, dtype=np.int8)
    if np.any((elevation < 0) | (elevation > 9)):
        raise ValueError("Les elevations doivent etre comprises entre 0 et 9.")

    return elevation


def validate_terrain(ascii_rows: list[str], elevation: np.ndarray) -> None:
    """
    Applique la regle d'arbitrage : une elevation 0 est infranchissable.
    Les rochers doivent donc avoir une elevation 0; une autre entite ne doit
    pas apparaitre sur une elevation 0.
    """
    for y, row in enumerate(ascii_rows):
        for x, tile in enumerate(row):
            level = int(elevation[y, x])
            if tile == "#" and level != 0:
                raise ValueError(
                    f"Rocher en ({x},{y}) avec elevation {level}; attendu 0."
                )
            if tile != "#" and level == 0:
                raise ValueError(
                    f"Elevation 0 en ({x},{y}) sous '{tile}'; "
                    "0 est reserve aux cases infranchissables."
                )

    # Contrainte garantie par le Twist pour trous et arbres.
    height, width = elevation.shape
    for y, row in enumerate(ascii_rows):
        for x, tile in enumerate(row):
            if tile not in {"o", "t"}:
                continue
            for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0)):
                nx, ny = x + dx, y + dy
                if 0 <= nx < width and 0 <= ny < height:
                    neighbor = int(elevation[ny, nx])
                    if neighbor > 0 and abs(int(elevation[y, x]) - neighbor) > 5:
                        raise ValueError(
                            f"Twist invalide en ({x},{y}) : '{tile}' a plus de "
                            "5 niveaux d'un voisin."
                        )


def import_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Impossible de charger {path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Look-ahead conditionnel : ne simuler les machines que si un coffre est
# devant le hero. Cette fonction est aussi l'autorite cote Python : si elle
# retourne False, les canaux 13 et 14 du tenseur d'observation restent a zero.
# `facing` est un tuple (dx, dy) : (0,-1)=UP, (0,1)=DOWN, (-1,0)=LEFT,
# (1,0)=RIGHT. Toute autre valeur (y compris (0,0)) desactive le lookahead.
# ---------------------------------------------------------------------------
FACING_VECTORS: dict[str, tuple[int, int]] = {
    "UP": (0, -1),
    "DOWN": (0, 1),
    "LEFT": (-1, 0),
    "RIGHT": (1, 0),
    "PUSH_UP": (0, -1),
    "PUSH_DOWN": (0, 1),
    "PUSH_LEFT": (-1, 0),
    "PUSH_RIGHT": (1, 0),
}


def should_run_lookahead(
    ascii_rows: list[str],
    hero_pos: tuple[int, int],
    facing: tuple[int, int],
) -> bool:
    """Retourne True ssi la case devant le hero contient un coffre '*'.

    Hors-carte ou direction invalide => False (pas de lookahead).
    Utilisee par l'environnement Python pour decider de remplir les canaux
    13/14. Mirroir de la regle cote Godot (HasChestAhead()).
    """
    if ascii_rows is None or not ascii_rows:
        return False
    dx, dy = facing
    if (dx, dy) == (0, 0):
        return False
    hx, hy = hero_pos
    fx, fy = hx + dx, hy + dy
    if fy < 0 or fy >= len(ascii_rows):
        return False
    row = ascii_rows[fy]
    if fx < 0 or fx >= len(row):
        return False
    return row[fx] == "*"


def parse_facing(action: str) -> tuple[int, int]:
    """Convertit un nom d'action en vecteur direction (dx, dy).

    UP/DOWN/LEFT/RIGHT et PUSH_* sont supportes. Toute autre action
    (HACK, WAIT, HACK_*) retourne (0, 0) -> lookahead desactive.
    """
    return FACING_VECTORS.get(action, (0, 0))


# ---------------------------------------------------------------------------
# CONTRAT DU MASQUE D'ACTIONS (mirror cote Godot Ikotofosa.cs / Imahaki.cs)
# ---------------------------------------------------------------------------
# Le flag `isOnEngine` (6eme scalaire, index 5) indique si le hero est
# actuellement en train de pirater une machine (RidingMachine != null).
#
# Sans ce masque, le modele peut rester coince dans une boucle infinie
# de HACK_CW / HACK_MOVE : il n'a jamais appris a emettre WAIT pour
# relacher la machine, et l'environnement ne le penalise pas pour les
# actions invalides.
#
# REGLE CONTRACTUELLE :
#   - isOnEngine = True (RidingMachine != null) :
#       VALIDES   = HACK_MOVE, HACK_FILL (F seulement), HACK_CW, HACK_CCW, WAIT
#       INVALIDES = UP, DOWN, LEFT, RIGHT, PUSH_*, HACK
#       (le hero est "absorbe" par la machine. WAIT = release.)
#   - isOnEngine = False (RidingMachine == null) :
#       VALIDES   = UP, DOWN, LEFT, RIGHT, WAIT, PUSH_*, HACK
#       INVALIDES = HACK_MOVE, HACK_FILL, HACK_CW, HACK_CCW
#       (le hero n'est pas sur une machine ; les HACK_* sont impossibles.)
#
# Cette fonction est l'autorite cote Python : l'environnement doit
# l'utiliser pour (1) penaliser les actions invalides et (2) exposer
# `action_mask` dans info pour les wrappers type MaskablePPO.
# ---------------------------------------------------------------------------
HACK_ACTIONS_F = {"HACK_MOVE", "HACK_FILL", "HACK_CW", "HACK_CCW"}
HACK_ACTIONS_M = {"HACK_MOVE", "HACK_CW", "HACK_CCW"}
OFF_ENGINE_ACTIONS = {
    "UP", "DOWN", "LEFT", "RIGHT", "WAIT",
    "PUSH_UP", "PUSH_DOWN", "PUSH_LEFT", "PUSH_RIGHT", "HACK",
}


def action_mask(is_on_engine: bool, action_names: list[str]) -> np.ndarray:
    """Retourne un tableau bool[n_actions] : True = action valide.

    Mirroir exact de Ikotofosa.BuildValidActionMask() / Imahaki.BuildValidActionMask().
    A utiliser par l'environnement pour penaliser les actions invalides et
    exposer `info["action_mask"]` pour MaskablePPO.
    """
    n = len(action_names)
    mask = np.zeros(n, dtype=bool)
    hack_set = HACK_ACTIONS_F if "HACK_FILL" in action_names else HACK_ACTIONS_M
    for i, name in enumerate(action_names):
        if is_on_engine:
            # Sur engine : seules HACK_* et WAIT sont valides.
            if name == "WAIT" or name in hack_set:
                mask[i] = True
        else:
            # Hors engine : MOVE/WAIT/PUSH/HACK valides, HACK_* interdits.
            if name in OFF_ENGINE_ACTIONS:
                mask[i] = True
    return mask


class ActionMasker(gym.Wrapper):
    """Wrapper qui expose `info["action_mask"]` calcule depuis `is_on_engine`.

    RecurrentPPO ne supporte pas nativement le masking, mais ce wrapper
    permet :
      - de journaliser les actions invalides tentees par le modele
      - de rester compatible avec MaskablePPO si l'utilisateur switch
      - de garantir que l'environnement sous-jacent a bien un contrat
        d'action_mask coherent avec cote Godot.

    L'environnement sous-jacent doit exposer `is_on_engine` dans info
    (ou implémenter sa propre methode `action_mask()`).
    """

    def __init__(self, env: gym.Env, action_names: list[str]):
        super().__init__(env)
        self.action_names = action_names

    def _mask_from_info(self, info: dict) -> np.ndarray:
        if "action_mask" in info and info["action_mask"] is not None:
            return np.asarray(info["action_mask"], dtype=bool)
        is_on = bool(info.get("is_on_engine", False))
        return action_mask(is_on, self.action_names)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        info["action_mask"] = self._mask_from_info(info)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        mask = self._mask_from_info(info)
        info["action_mask"] = mask
        # Journalisation : si le modele a emis une action invalide, on
        # l'indique dans info pour que EpisodeStatsCallback puisse la compter.
        a = int(action)
        if 0 <= a < len(mask) and not mask[a]:
            info["invalid_action_attempted"] = True
        return obs, reward, terminated, truncated, info


def find_factory(module) -> Callable[..., gym.Env]:
    for name in ("make_env", "AlgoEnv", "AlgoGamesEnv"):
        factory = getattr(module, name, None)
        if callable(factory):
            return factory
    raise ImportError(
        "AlgoEnv.py doit exposer make_env, AlgoEnv ou AlgoGamesEnv."
    )


def filter_supported_kwargs(factory: Callable[..., Any], kwargs: dict) -> dict:
    """Evite de casser une ancienne signature d'environnement."""
    signature = inspect.signature(factory)
    if any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        return kwargs

    aliases = {
        "map_path": ("map_file", "map"),
        "elevation_path": ("elevation_file", "elevation"),
        "preprocessor_path": ("preprocessor_file", "preprocessor"),
        "hero": ("hero_id", "agent"),
    }

    accepted = set(signature.parameters)
    result: dict[str, Any] = {}
    for key, value in kwargs.items():
        if key in accepted:
            result[key] = value
            continue
        for alias in aliases.get(key, ()):
            if alias in accepted:
                result[alias] = value
                break
    return result


class ContractEnv(gym.Wrapper):
    """
    Verifie le contrat d'observation exige par le modele/C# sans modifier
    la logique de l'environnement existant. L'espace d'actions est impose
    par le hero selectionne (Discrete(14) pour F, Discrete(13) pour M).
    """

    def __init__(self, env: gym.Env, hero: str):
        super().__init__(env)
        if hero not in HEROES:
            raise ValueError(f"Hero inconnu : {hero!r}. Attendu : {list(HEROES)}.")
        self.hero = hero
        self.action_space = spaces.Discrete(HEROES[hero]["n_actions"])
        self.action_names = HEROES[hero]["actions"]
        self.observation_space = spaces.Dict(
            {
                "grid": spaces.Box(
                    low=-1.0,
                    high=1.0,
                    shape=(N_GRID_CHANNELS, MAX_HEIGHT, MAX_WIDTH),
                    dtype=np.float32,
                ),
                "scalars": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(N_SCALARS,),
                    dtype=np.float32,
                ),
            }
        )

    @staticmethod
    def _normalize_observation(observation: Any) -> dict[str, np.ndarray]:
        if not isinstance(observation, dict):
            raise TypeError("L'observation doit etre un dictionnaire.")

        grid = observation.get("grid", observation.get("map"))
        scalars = observation.get("scalars", observation.get("stats"))
        if grid is None or scalars is None:
            raise KeyError("L'observation doit contenir grid/scalars.")

        grid = np.asarray(grid, dtype=np.float32)
        scalars = np.asarray(scalars, dtype=np.float32)

        if grid.shape != (N_GRID_CHANNELS, MAX_HEIGHT, MAX_WIDTH):
            raise ValueError(
                f"Forme grid invalide {grid.shape}; attendu "
                f"({N_GRID_CHANNELS}, {MAX_HEIGHT}, {MAX_WIDTH})."
            )
        if scalars.shape != (N_SCALARS,):
            raise ValueError(
                f"Forme scalars invalide {scalars.shape}; "
                f"attendu ({N_SCALARS},)."
            )
        if not np.isfinite(grid).all() or not np.isfinite(scalars).all():
            raise ValueError("L'observation contient NaN ou Inf.")

        return {
            "grid": np.ascontiguousarray(grid),
            "scalars": np.clip(scalars, 0.0, 1.0),
        }

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        return self._normalize_observation(observation), info

    def step(self, action):
        # SB3 fournit un scalaire np.int64 ; on transmet tel quel a l'env sous-jacent.
        observation, reward, terminated, truncated, info = self.env.step(int(action))

        # L'environnement doit signaler la cause prioritaire dans info.
        if info.get("resources_exhausted"):
            terminated = True
            truncated = False
            info["termination_reason"] = "resources_exhausted"
        elif truncated:
            info.setdefault("termination_reason", "timeout")

        info.setdefault("hero", self.hero)
        info.setdefault("action_name", self.action_names[int(action)])

        return (
            self._normalize_observation(observation),
            float(reward),
            bool(terminated),
            bool(truncated),
            info,
        )


class AlgoGamesEnv(gym.Env):
    """Environnement AlgoGames minimal embarque dans AlgoTrain.

    Ce squelette respecte le contrat d'observation attendu par le trainer.
    Il n'implemente pas toute la mecanique du jeu (collisions, push, etc.)
    mais il respecte DEJA le contrat du flag `isOnEngine` :
      - suit les transitions HACK / WAIT / HACK_* pour mettre a jour l'etat
      - penalise les actions invalides (reward = invalid_action)
      - expose `is_on_engine` et `action_mask` dans info
    Cela permet au modele d'etre entraene sur des observations ou isOnEngine
    vaut tantot 0 tantot 1, evitant ainsi la boucle infinie HACK_CW/HACK_MOVE
    a l'inference.

    Remplacez / etendez avec les vraies regles du GDD quand le moteur est
    disponible. Le point critique est que l'espace d'actions (Discrete(14)
    ou Discrete(13)) doit matcher exactement ce que Exports.py produira
    comme ONNX, et que le contrat du masque d'actions doit rester identique
    a `action_mask(is_on_engine, action_names)`.
    """

    # Penalties utilisees quand l'utilisateur ne fournit pas reward_config.
    _DEFAULT_INVALID_PENALTY = -0.15
    _DEFAULT_IDLE_PENALTY = -0.02
    _DEFAULT_USEFUL_HACK_REWARD = 0.10
    _MAX_BATTERY = 100.0
    _MAX_STAMINA = 100.0

    def __init__(
        self,
        map_path: str,
        elevation_path: str,
        hero: str = "F",
        preprocessor_path: str | None = None,
        augmentation: str = "identity",
        augmentations: tuple = AUGMENTATIONS,
        engine_config: dict | None = None,
        reward_config: dict | None = None,
        seed: int = 0,
        max_height: int = MAX_HEIGHT,
        max_width: int = MAX_WIDTH,
        grid_channels: int = N_GRID_CHANNELS,
        scalar_count: int = N_SCALARS,
    ) -> None:
        super().__init__()
        if hero not in HEROES:
            raise ValueError(f"hero inconnu : {hero!r}. Attendu : {list(HEROES)}.")
        self.hero = hero
        self.hero_name = HEROES[hero]["name"]
        self.n_actions = HEROES[hero]["n_actions"]
        self.action_names = HEROES[hero]["actions"]
        self.machine_target = HEROES[hero]["machine_target"]

        self.map_path = Path(map_path)
        self.elevation_path = Path(elevation_path)
        self.preprocessor_path = (
            Path(preprocessor_path) if preprocessor_path is not None else None
        )
        self.augmentation = augmentation
        self.augmentations = augmentations
        self.engine_config = engine_config or {}
        self.reward_config = reward_config or {}
        self.seed = seed
        self.max_height = max_height
        self.max_width = max_width
        self.grid_channels = grid_channels
        self.scalar_count = scalar_count

        # Validation des entrees via les helpers partages.
        self.height, self.width, self.max_time, self.ascii_rows = read_map_header(self.map_path)
        self.elevation = read_elevation(self.elevation_path, self.height, self.width)
        validate_terrain(self.ascii_rows, self.elevation)

        # Preprocesseur optionnel (legacy).
        if self.preprocessor_path is not None:
            try:
                self.preprocessor = import_module(
                    self.preprocessor_path, "algogames_map_preprocessor"
                )
            except Exception:
                self.preprocessor = None
        else:
            self.preprocessor = None

        # Contrat d'observation identique pour les deux heroes.
        self.observation_space = spaces.Dict(
            {
                "grid": spaces.Box(
                    low=-1.0,
                    high=1.0,
                    shape=(self.grid_channels, MAX_HEIGHT, MAX_WIDTH),
                    dtype=np.float32,
                ),
                "scalars": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(self.scalar_count,),
                    dtype=np.float32,
                ),
            }
        )

        # Action space CONTRACTUEL : 14 pour Ikotofosa, 13 pour Imahaki.
        self.action_space = spaces.Discrete(self.n_actions)

        self.rng = random.Random(seed)
        self._state = None
        # Etat hack : tracking de isOnEngine pour le contrat du masque d'actions.
        self._is_on_engine = False
        self._battery = self._MAX_BATTERY
        self._stamina = self._MAX_STAMINA
        self._tick = 0
        self._hack_action_count = 0

    # ------------------------------------------------------------------
    # Contrat du masque d'actions : mirror cote Godot (BuildValidActionMask).
    # L'environnement concret pourra surcharger cette methode pour ajouter
    # des contraintes specifiques (ex : HACK valide seulement si une machine
    # est sur la case), mais le contrat de base DOIT etre respecte.
    # ------------------------------------------------------------------
    def action_mask(self) -> np.ndarray:
        return action_mask(self._is_on_engine, self.action_names)

    def reset(self, **kwargs):
        grid = np.zeros((self.grid_channels, MAX_HEIGHT, MAX_WIDTH), dtype=np.float32)
        scalars = np.zeros((self.scalar_count,), dtype=np.float32)

        # Random start : 30% du temps, le hero demarre deja sur engine.
        # Cela force le modele a voir isOnEngine=1 pendant l'entrainement
        # et a apprendre a emettre WAIT pour relacher la machine.
        self._is_on_engine = self.rng.random() < 0.30
        self._battery = self._MAX_BATTERY
        self._stamina = self._MAX_STAMINA
        self._tick = 0
        self._hack_action_count = 0

        # Mettre a jour les scalaires : stamina, batterie, temps, X, Y, isOnEngine
        scalars[0] = self._stamina / self._MAX_STAMINA
        scalars[1] = self._battery / self._MAX_BATTERY
        scalars[2] = 1.0  # temps restant (full au reset)
        scalars[3] = 0.5  # position X normalisee (centre)
        scalars[4] = 0.5  # position Y normalisee (centre)
        scalars[5] = 1.0 if self._is_on_engine else 0.0

        self._state = {"grid": grid, "scalars": scalars}
        info = {
            "hero": self.hero,
            "hero_name": self.hero_name,
            "n_actions": self.n_actions,
            "is_on_engine": self._is_on_engine,
            "action_mask": self.action_mask(),
        }
        return self._state, info

    def step(self, action):
        a = int(action)
        action_name = self.action_names[a] if 0 <= a < self.n_actions else "WAIT"
        mask = self.action_mask()
        is_invalid = not (0 <= a < self.n_actions and mask[a])

        invalid_penalty = float(
            self.reward_config.get("invalid_action", self._DEFAULT_INVALID_PENALTY)
        )
        idle_penalty = float(
            self.reward_config.get("idle_action", self._DEFAULT_IDLE_PENALTY)
        )
        useful_hack_reward = float(
            self.reward_config.get("useful_hack", self._DEFAULT_USEFUL_HACK_REWARD)
        )

        reward = 0.0
        terminated = False
        truncated = False

        if is_invalid:
            # Action invalide pour l'etat courant : penalite, etat invariant.
            reward = invalid_penalty
            self._hack_action_count = 0  # reset streak
        else:
            # Transition d'etat selon l'action valide.
            if action_name == "HACK":
                # HACK valide seulement hors engine -> passe sur engine.
                self._is_on_engine = True
                self._battery = max(0.0, self._battery - 1.0)
                self._hack_action_count = 0
                reward = 0.0  # l'environnement reel peut donner useful_hack
            elif action_name == "WAIT":
                # WAIT hors engine = idle. WAIT sur engine = release.
                if self._is_on_engine:
                    self._is_on_engine = False
                    self._hack_action_count = 0
                    reward = useful_hack_reward * 0.5  # demi-recompense (squelette)
                else:
                    reward = idle_penalty
            elif action_name in ("HACK_MOVE", "HACK_FILL", "HACK_CW", "HACK_CCW"):
                # HACK_* valide seulement sur engine, consomme batterie.
                self._battery = max(0.0, self._battery - 1.0)
                self._hack_action_count += 1
                # Petite recompense pour encourager les HACK_* (utiles).
                reward = useful_hack_reward
                # Securite anti-boucle infinie : si plus de batterie, forcer
                # la sortie au prochain tick (l'environnement reel devrait
                # s'occuper de ca via le GDD ; ici on tronque par securite).
                if self._battery <= 0.0:
                    self._is_on_engine = False
                    self._hack_action_count = 0
            elif action_name in ("UP", "DOWN", "LEFT", "RIGHT"):
                # Deplacement valide hors engine : cout stamina standard.
                self._stamina = max(0.0, self._stamina - 1.0)
                self._hack_action_count = 0
                reward = 0.0
            elif action_name.startswith("PUSH_"):
                # Push valide hors engine : cout stamina plus eleve.
                self._stamina = max(0.0, self._stamina - 1.0)
                self._hack_action_count = 0
                reward = 0.0

        # Avance du temps.
        self._tick += 1
        if self._tick >= self.max_time:
            truncated = True

        # Epuisement des ressources -> fin d'episode.
        if self._battery <= 0.0 and self._is_on_engine:
            # Batterie videe en hackant : on force la sortie (deja fait plus haut).
            pass
        if self._stamina <= 0.0:
            terminated = True

        # Mettre a jour les scalaires de l'observation.
        scalars = np.copy(self._state["scalars"])
        scalars[0] = self._stamina / self._MAX_STAMINA
        scalars[1] = self._battery / self._MAX_BATTERY
        scalars[2] = max(0.0, 1.0 - self._tick / max(1, self.max_time))
        scalars[5] = 1.0 if self._is_on_engine else 0.0

        obs = {
            "grid": np.copy(self._state["grid"]),
            "scalars": scalars,
        }
        self._state["scalars"] = scalars

        info = {
            "hero": self.hero,
            "action_name": action_name,
            "is_on_engine": self._is_on_engine,
            "action_mask": self.action_mask(),
            "invalid_action": is_invalid,
        }
        return obs, float(reward), bool(terminated), bool(truncated), info


class EpisodeStatsCallback(BaseCallback):
    """Journalise par hero : episodes, actions invalides, hacks utiles, etc.

    Compte notamment les `invalid_action_attempted` signales par ActionMasker
    pour que l'utilisateur puisse detecter en TensorBoard si le modele tente
    d'emettre des actions invalides (ex : HACK_CW alors que isOnEngine=0).
    """

    def __init__(self, hero: str):
        super().__init__()
        self.hero = hero
        self.episodes = 0
        self.invalid_attempts = 0

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            episode = info.get("episode")
            if episode is not None:
                self.episodes += 1
                self.logger.record(f"{self.hero}/episodes", self.episodes)

            # ActionMasker signale les tentatives d'actions invalides.
            if info.get("invalid_action_attempted") or info.get("invalid_action"):
                self.invalid_attempts += 1
                self.logger.record(f"{self.hero}/invalid_action_attempts",
                                   float(self.invalid_attempts))

            for key in (
                "stones_collected",
                "chests_hidden",
                "chests_destroyed",
                "invalid_actions",
                "blocked_ticks",
                "useful_hacks",
                "useful_fills",
                "useful_cuts",
            ):
                if key in info:
                    self.logger.record(f"{self.hero}/{key}", float(info[key]))
        return True


def choose_augmentation(mode: str, rank: int) -> str:
    if mode == "all":
        return AUGMENTATIONS[rank % len(AUGMENTATIONS)]
    return mode


def make_single_env(
    factory: Callable[..., gym.Env],
    map_path: Path,
    elevation_path: Path,
    hero: str,
    augmentation: str,
    seed: int,
    rank: int,
) -> Callable[[], gym.Env]:
    def initializer() -> gym.Env:
        kwargs = {
            "map_path": str(map_path),
            "elevation_path": str(elevation_path),
            "hero": hero,
            "augmentation": choose_augmentation(augmentation, rank),
            "augmentations": AUGMENTATIONS,
            "engine_config": dict(DEFAULT_ENGINE_CONFIG),
            "reward_config": dict(DEFAULT_REWARD_CONFIG),
            "seed": seed + rank,
            "max_height": MAX_HEIGHT,
            "max_width": MAX_WIDTH,
            "grid_channels": N_GRID_CHANNELS,
            "scalar_count": N_SCALARS,
        }
        env = factory(**filter_supported_kwargs(factory, kwargs))
        env = ContractEnv(env, hero=hero)
        # Wrap avec ActionMasker pour exposer info["action_mask"] et
        # journaliser les actions invalides. C'est le contrat mirror
        # de BuildValidActionMask() cote Godot : sans lui, un modele
        # mal entraene peut rester coince dans une boucle infinie de
        # HACK_CW / HACK_MOVE a l'inference.
        env = ActionMasker(env, action_names=HEROES[hero]["actions"])
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env

    return initializer


def normalize_output(path: str, hero: str) -> Path:
    """Normalise le chemin de sortie et ajoute un suffixe hero si absent.

    Convention : si l'utilisateur ne donne pas deja un nom contenant
    'ikotofosa' ou 'imahaki', on ajoute le suffixe _<hero> pour eviter
    d'ecraser un modele de l'autre hero.
    """
    output = Path(path).expanduser().resolve()
    if output.suffix != ".zip":
        output = output.with_suffix(".zip")

    hero_name_lower = HEROES[hero]["name"].lower()
    stem_lower = output.stem.lower()
    if hero_name_lower not in stem_lower and hero.lower() not in stem_lower:
        output = output.with_name(f"{output.stem}_{hero}{output.suffix}")

    output.parent.mkdir(parents=True, exist_ok=True)
    return output


def build_policy_kwargs(n_actions: int) -> dict:
    """Construit les policy_kwargs en fonction du nombre d'actions.

    La taille de la tete d'action depend de n_actions ; SB3 la deduit
    automatiquement de l'espace d'actions. On garde net_arch leger pour
    que les deux politiques convergent avec le meme budget d'hyperparams.
    """
    return {
        "features_extractor_class": GridScalarExtractor,
        "features_extractor_kwargs": {"features_dim": 128},
        "lstm_hidden_size": 128,
        "n_lstm_layers": 1,
        "net_arch": {"pi": [64], "vf": [64]},
    }


def main() -> None:
    args = parse_args()
    hero = args.hero
    n_actions = HEROES[hero]["n_actions"]

    map_path = resolve_file(args.map_path, "map.txt")
    elevation_path = resolve_file(args.elevation_path, "elevation.txt")
    output_path = normalize_output(args.output, hero)

    height, width, max_time, ascii_rows = read_map_header(map_path)
    elevation = read_elevation(elevation_path, height, width)
    validate_terrain(ascii_rows, elevation)

    # On utilise l'environnement embarque (squelette) ; remplacable par
    # AlgoEnv.py quand le moteur concret sera disponible.
    factory = AlgoGamesEnv

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.n_envs < 1:
        raise ValueError("--n-envs doit etre superieur ou egal a 1.")
    if args.n_steps < 2:
        raise ValueError("--n-steps doit etre superieur ou egal a 2.")

    print(
        f"[train] hero={HEROES[hero]['name']} ({hero}) "
        f"n_actions={n_actions} "
        f"actions={HEROES[hero]['actions']}"
    )

    env_fns = [
        make_single_env(
            factory,
            map_path,
            elevation_path,
            hero,
            args.augmentation,
            args.seed,
            rank,
        )
        for rank in range(args.n_envs)
    ]

    if args.check_env:
        candidate = env_fns[0]()
        try:
            check_env(candidate, warn=True)
        finally:
            candidate.close()

    train_env = VecMonitor(DummyVecEnv(env_fns))

    eval_env = VecMonitor(
        DummyVecEnv(
            [
                make_single_env(
                    factory,
                    map_path,
                    elevation_path,
                    hero,
                    "identity",
                    args.seed + 100_000,
                    0,
                )
            ]
        )
    )

    policy_kwargs = build_policy_kwargs(n_actions)

    rollout_size = args.n_steps * args.n_envs
    batch_size = min(args.batch_size, rollout_size)
    while rollout_size % batch_size != 0 and batch_size > 1:
        batch_size -= 1

    if args.resume:
        resume_path = resolve_file(args.resume, "Modele a reprendre")
        model = RecurrentPPO.load(
            str(resume_path),
            env=train_env,
            device=args.device,
        )
    else:
        model = RecurrentPPO(
            policy=MultiInputLstmPolicy,
            env=train_env,
            device=args.device,
            policy_kwargs=policy_kwargs,
            verbose=1,
            seed=args.seed,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=batch_size,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            ent_coef=args.ent_coef,
            tensorboard_log=str(output_path.parent / f"tensorboard_{hero}"),
        )

    checkpoint_dir = output_path.parent / f"checkpoints_{hero}"
    best_dir = output_path.parent / f"best_{hero}"
    checkpoint_dir.mkdir(exist_ok=True)
    best_dir.mkdir(exist_ok=True)

    callbacks: list[BaseCallback] = [EpisodeStatsCallback(hero=hero)]

    if args.checkpoint_freq > 0:
        callbacks.append(
            CheckpointCallback(
                save_freq=max(args.checkpoint_freq // args.n_envs, 1),
                save_path=str(checkpoint_dir),
                name_prefix=output_path.stem,
                save_replay_buffer=False,
                save_vecnormalize=False,
            )
        )

    if args.eval_freq > 0:
        callbacks.append(
            EvalCallback(
                eval_env,
                best_model_save_path=str(best_dir),
                log_path=str(best_dir),
                eval_freq=max(args.eval_freq // args.n_envs, 1),
                n_eval_episodes=5,
                deterministic=True,
                render=False,
            )
        )

    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=CallbackList(callbacks),
            reset_num_timesteps=not bool(args.resume),
            progress_bar=True,
        )
        model.save(str(output_path.with_suffix("")))
        print(f"[train] modele sauvegarde : {output_path}")
        print(
            f"[train] hero={HEROES[hero]['name']} carte={height}x{width} "
            f"temps={max_time} "
            f"observation=({N_GRID_CHANNELS}, {MAX_HEIGHT}, {MAX_WIDTH})+"
            f"{N_SCALARS} actions={n_actions}"
        )
        print(
            f"[train] actions : {HEROES[hero]['actions']}"
        )
    finally:
        train_env.close()
        eval_env.close()


if __name__ == "__main__":
    main()
