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

Observation conforme au modele existant et au preprocesseur C# cible :
    grid    : float32[13, 30, 30]
              11 canaux one-hot + elevation absolue + elevation relative
    scalars : float32[6]
              stamina, batterie, temps, position X/Y, isOnEngine

Le script peut etre appele deux fois (une fois par hero) :

    python AlgoTrain.py --hero F --map map.txt --elevation elevation.txt --output ModelStates/ikotofosa
    python AlgoTrain.py --hero M --map map.txt --elevation elevation.txt --output ModelStates/imahaki

Puis Exports.py produira ikotofosa.onnx (14 actions) et imahaki.onnx (13 actions).

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
N_GRID_CHANNELS = 13
N_SCALARS = 6

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
    Il n'implemente pas la mecanique du jeu : remplacez / etendez avec les
    vraies regles du GDD quand le moteur est disponible. Le point critique
    est que l'espace d'actions (Discrete(14) ou Discrete(13)) doit matcher
    exactement ce que Exports.py produira comme ONNX.
    """

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

    def reset(self, **kwargs):
        grid = np.zeros((self.grid_channels, MAX_HEIGHT, MAX_WIDTH), dtype=np.float32)
        scalars = np.zeros((self.scalar_count,), dtype=np.float32)
        self._state = {"grid": grid, "scalars": scalars}
        return self._state, {
            "hero": self.hero,
            "hero_name": self.hero_name,
            "n_actions": self.n_actions,
        }

    def step(self, action):
        # Pas de regles de jeu implementees : observation stable zero.
        obs = {
            "grid": np.copy(self._state["grid"]),
            "scalars": np.copy(self._state["scalars"]),
        }
        reward = 0.0
        terminated = False
        truncated = False
        info = {
            "hero": self.hero,
            "action_name": self.action_names[int(action)],
        }
        return obs, float(reward), bool(terminated), bool(truncated), info


class EpisodeStatsCallback(BaseCallback):
    """Journalise par hero : episodes, actions invalides, hacks utiles, etc."""

    def __init__(self, hero: str):
        super().__init__()
        self.hero = hero
        self.episodes = 0

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            episode = info.get("episode")
            if episode is not None:
                self.episodes += 1
                self.logger.record(f"{self.hero}/episodes", self.episodes)

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
