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
from datetime import datetime
import importlib.util
import inspect
import json
import os
import random
from pathlib import Path
from typing import Any, Callable

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from sb3_contrib import MaskablePPO                                      # MIGRATION MASKABLEPPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback   # MIGRATION MASKABLEPPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
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

# Dossier dedie, partage entre Optuna.py (ecriture) et AlgoTrain.py (lecture)
# pour les hyperparametres trouves par Optuna. Un fichier par hero pour
# pouvoir lancer F et M en parallele sans collision.
HYPERPARAMS_DIR = Path("OptunaParams")


def hyperparams_path(hero: str) -> Path:
    HYPERPARAMS_DIR.mkdir(parents=True, exist_ok=True)
    return HYPERPARAMS_DIR / f"best_hyperparams_{hero}.json"


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
    "hero_uphill_block": 3,
    "hero_downhill_block": 5,
    "machine_height_block": 5,
    "machine_chest_uphill_block": 3,
    "uphill_stamina_cost": 3.0,
    "downhill_stamina_cost": 0.0,
    "forbid_chest_uphill": True,
    "destroy_chest_drop": 5,
    "resource_exhaustion_first": True,
}

# Les rewards d'apprentissage sont strictement alignes sur les composantes du
# score officiel :
#
#   R = C*150 + P*25 + (S+B)//4 + T//2
#
# Les gains terminaux exacts ne sont pas redistribues ici. Le shaping sert
# uniquement a guider l'agent vers une amelioration potentielle du score :
# rapprochement d'un objectif, conservation des ressources et prevention de
# la perte d'un coffre.
DEFAULT_REWARD_CONFIG = {
    # Evenements qui augmentent directement le score officiel.
    "stone_collected": 25.0,
    "chest_hidden": 150.0,
    # Progression vers les deux objectifs officiels.
    "progress_to_stone": 0.30,
    "regress_from_stone": -0.30,
    "progress_to_chest": 0.40,
    "regress_from_chest": -0.40,
    "progress_chest_to_bush": 0.75,
    "regress_chest_from_bush": -0.75,
    # Une action de machine n'est recompensee que si elle rapproche le hero
    # d'une pierre, d'un coffre, ou ouvre un chemin vers un de ces objectifs.
    "useful_hack": 0.20,
    "useful_fill": 0.50,
    "useful_cut": 0.50,
    # Preservation des composantes S et B du score final.
    # Le cout normal de stamina/batterie est deja visible dans le score :
    # aucune penalite generique supplementaire n'est appliquee aux actions
    # valides. Seul le gaspillage manifeste est penalise.
    "wasted_battery": -0.30,
    "resource_exhausted": -5.0,
    # Actions invalides ou inutiles.
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
    # Perte d'un objectif du score.
    "chest_destroyed": -25.0,
    "voluntary_chest_loss": -50.0,
    # Le timeout ne doit pas ajouter une penalite arbitraire : le temps restant
    # T est deja inclus exactement dans le bonus terminal du score officiel.
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

        # cf. diagnostic Cause 2 : AdaptiveAvgPool2d((1,1)) ecrasait chaque
        # canal 30x30 en une seule moyenne, detruisant la position relative
        # des objectifs (coffre a gauche/droite du hero, etc.) -> aliasing
        # observationnel ou WAIT devient la meilleure action "moyenne"
        # observable. On conserve desormais une grille spatiale 5x5.
        self.cnn = nn.Sequential(
            nn.Conv2d(N_GRID_CHANNELS, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((5, 5)),
            nn.Flatten(),
        )

        self.mlp = nn.Sequential(
            nn.Linear(N_SCALARS, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )

        self.combined = nn.Sequential(
            nn.Linear(64 * 5 * 5 + 32, features_dim),
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
    # --- Optimisations CPU ---
    # RecurrentPPO + LSTM 128 est lourd sur CPU. Pour accelerer :
    #   --lstm-size 64     (LSTM plus petit, ~2x plus rapide)
    #   --no-eval          (desactive EvalCallback, ~30% plus rapide)
    #   --no-progress-bar  (desactive tqdm, ~5% plus rapide)
    # LSTM size removed: MaskablePPO is feedforward (no LSTM).      # MIGRATION MASKABLEPPO
    parser.add_argument(
        "--no-eval", action="store_true",
        help="Desactive EvalCallback (recommande sur CPU).",
    )
    parser.add_argument(
        "--no-progress-bar", action="store_true",
        help="Desactive la barre de progression tqdm (overhead CPU).",
    )
    # --- Logging de l'evolution de la map ---
    parser.add_argument(
        "--log-dir", default="TrainingLogs",
        help="Repertoire de sortie pour les logs de session (un fichier .log par session).",
    )
    parser.add_argument(
        "--log-episodes", type=int, default=50,
        help="Log l'evolution de la map tous les N episodes (default 50).",
    )
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

    # REGLE : l'elevation est chargee une fois pour toutes depuis le fichier.
    # Elle NE DOIT JAMAIS evoluer pendant l'entrainement (sinon les regles
    # du Twist - delta > 2 = bloque - perdent leur stabilite). On rend donc
    # le ndarray readonly : toute tentative d'ecriture levera une erreur
    # runtime, ce qui detecte immediatement un bug de mutation accidentelle.
    elevation.setflags(write=False)
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
        # Cache mis a jour a chaque reset()/step() ; c'est ce cache que
        # action_masks() (voir plus bas) renvoie. Valeur par defaut avant
        # le premier reset() : tout autorise, jamais utilisee en pratique
        # puisque SB3 appelle toujours reset() avant collect_rollouts().
        self._current_mask = np.ones(len(action_names), dtype=bool)

    def _mask_from_info(self, info: dict) -> np.ndarray:
        if "action_mask" in info and info["action_mask"] is not None:
            return np.asarray(info["action_mask"], dtype=bool)
        is_on = bool(info.get("is_on_engine", False))
        return action_mask(is_on, self.action_names)

    # -----------------------------------------------------------------
    # MIGRATION MASKABLEPPO : contrat officiel MaskableEnv de sb3_contrib.
    # sb3_contrib.common.maskable.utils.is_masking_supported()/
    # get_action_masks() cherchent une methode nommee EXACTEMENT
    # `action_masks` (pluriel, sans argument). Ni `info["action_mask"]`
    # ni une methode `action_mask()` (singulier, cf. AlgoGamesEnv) ne
    # remplissent ce contrat -> c'est la cause de
    # "ValueError: Environment does not support action masking."
    # On expose donc le cache ici ; DummyVecEnv/VecMonitor le trouveront
    # via env_method("action_masks") en traversant les gym.Wrapper
    # (Monitor -> ActionMasker) grace au __getattr__ standard de Wrapper.
    # -----------------------------------------------------------------
    def action_masks(self) -> np.ndarray:
        return self._current_mask

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._current_mask = self._mask_from_info(info)
        info["action_mask"] = self._current_mask
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        mask = self._mask_from_info(info)
        self._current_mask = mask
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
    """Environnement AlgoGames fonctionnel embarque dans AlgoTrain.

    IMPLEMENTATION :
      - Encode la carte ASCII dans les 15 canaux du tenseur grid :
          0  : floor (.)
          1  : wall (#)
          2  : chest (*)
          3  : hole (o)
          4  : tree (t)
          5  : hidden_chest (@)
          6  : stone (+)
          7  : excavator (X)
          8  : grappler (G)
          9  : hero (F ou M) - one-hot a la position courante du hero
          10 : hero_facing (one-hot a la position devant le hero)
          11 : elevation absolue (normalisee [0,1] -> [-1,1])
          12 : elevation relative (level - mean, clipped [-1,1])
          13 : next_X (look-ahead excavateur)
          14 : next_G (look-ahead grappler)
      - Deplace reellement le hero sur MOVE (avec collisions terrain)
      - Pousse les coffres sur PUSH (avec collision check)
      - Suit les transitions HACK / WAIT / HACK_* pour isOnEngine
      - Recompenses : move valide (+), wall hit (-), idle (-), stone collecte (++),
        chest hidden (+++), chest destroyed (---), invalid action (--)
      - Buffers pre-alloues pour eviter GC pressure sur CPU

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
    _DEFAULT_WALL_HIT_PENALTY = -0.05
    _DEFAULT_MOVE_REWARD = 0.02
    _DEFAULT_STONE_REWARD = 1.0
    _DEFAULT_CHEST_HIDDEN_REWARD = 5.0
    _DEFAULT_CHEST_DESTROYED_PENALTY = -10.0
    _MAX_BATTERY = 100.0
    _MAX_STAMINA = 100.0
    _PUSH_STAMINA_COST = 5.0
    _MOVE_STAMINA_COST = 1.0
    _UPHILL_STAMINA_COST = 3.0

    # Mapping ASCII -> canal one-hot (cf. InputPreprocessor.cs cote Godot).
    _TILE_CHANNELS: dict[str, int] = {
        ".": 0,   # floor
        "#": 1,   # wall
        "*": 2,   # chest
        "o": 3,   # hole
        "t": 4,   # tree
        "@": 5,   # hidden_chest
        "+": 6,   # stone
        "X": 7,   # excavator
        "G": 8,   # grappler
    }
    # F et M sont encodes dans le canal 9 (hero) - dynamique selon le hero.
    _HERO_CHANNEL = 9
    _HERO_FACING_CHANNEL = 10
    _ABS_ELEV_CHANNEL = 11
    _REL_ELEV_CHANNEL = 12
    _NEXT_X_CHANNEL = 13
    _NEXT_G_CHANNEL = 14

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
        # IMPORTANT : self.elevation est readonly (setflags write=False dans
        # read_elevation) pour garantir qu'elle n'evolue jamais pendant
        # l'entrainement. C'est un contrat du Twist.
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

        # --- Buffers pre-alloues pour eviter GC pressure sur CPU ---
        # Ces buffers sont reutilises a chaque step ; on ne les alloue qu'une
        # fois. La GC pressure etait la cause principale du ralentissement
        # observe apres quelques milliers de steps (15*30*30 = 13,500 floats
        # alloues + copies a chaque step => ~50 KB/step => 50 MB/1000 steps
        # de dechets a ramasser).
        self._grid_buf = np.zeros((self.grid_channels, MAX_HEIGHT, MAX_WIDTH), dtype=np.float32)
        self._scalars_buf = np.zeros((self.scalar_count,), dtype=np.float32)
        self._mask_buf = np.zeros(self.n_actions, dtype=bool)

        # Encode la map statique une seule fois dans _static_grid (canals 0-8).
        # Les canaux 9 (hero), 10 (facing), 11-12 (elevation) sont statiques
        # aussi, on les encode aussi ici. Seuls 9, 10, 13, 14 peuvent changer
        # pendant un episode (position du hero + look-ahead machines).
        self._static_grid = np.zeros((self.grid_channels, MAX_HEIGHT, MAX_WIDTH), dtype=np.float32)
        self._encode_static_grid()

        # Etat dynamique du hero.
        self._hero_x, self._hero_y = 0, 0
        self._hero_facing = (1, 0)  # RIGHT par defaut
        self._is_on_engine = False
        self._battery = self._MAX_BATTERY
        self._stamina = self._MAX_STAMINA
        self._tick = 0
        self._hack_action_count = 0
        self._stones_collected = 0
        self._chests_hidden = 0
        self._chests_destroyed = 0
        self._invalid_streak = 0

        # Positions dynamiques des coffres (set de (x,y)) - peut evoluer.
        self._chest_positions: set[tuple[int, int]] = set()
        self._stone_positions: set[tuple[int, int]] = set()
        self._machine_positions: set[tuple[int, int]] = set()
        self._scan_dynamic_entities()

    # ------------------------------------------------------------------
    # Encodage du tenseur grid : appele une fois dans __init__ pour la
    # partie statique, puis a chaque reset/step pour la partie dynamique.
    # ------------------------------------------------------------------
    def _encode_static_grid(self):
        """Encode les canaux statiques 0-8 (tuiles ASCII) et 11-12 (elevation).
        Les canaux 9 (hero), 10 (facing), 13-14 (look-ahead) restent a zero
        et seront remplis dynamiquement dans _encode_dynamic_grid."""
        # Reset static portion (au cas ou on re-encode apres une mutation).
        self._static_grid[:9] = 0.0
        self._static_grid[11:15] = 0.0

        # Canaux 0-8 : one-hot par tuile ASCII.
        for y, row in enumerate(self.ascii_rows):
            for x, c in enumerate(row):
                if c in self._TILE_CHANNELS:
                    ch = self._TILE_CHANNELS[c]
                    self._static_grid[ch, y, x] = 1.0
                # Le hero F/M est stocke dans le canal 9 (dynamique), pas ici.

        # Canal 11 : elevation absolue normalisee [-1, 1].
        # elevation 0..9 -> -1..+1
        elev_norm = (self.elevation.astype(np.float32) / 4.5) - 1.0
        self._static_grid[self._ABS_ELEV_CHANNEL, :self.height, :self.width] = elev_norm

        # Canal 12 : elevation relative = level - mean, clipped [-1, 1].
        mean_elev = float(self.elevation.mean())
        rel_elev = np.clip(
            (self.elevation.astype(np.float32) - mean_elev) / 4.5,
            -1.0, 1.0,
        )
        self._static_grid[self._REL_ELEV_CHANNEL, :self.height, :self.width] = rel_elev

    def _scan_dynamic_entities(self):
        """Scan la map pour initialiser les sets de positions dynamiques."""
        self._chest_positions.clear()
        self._stone_positions.clear()
        self._machine_positions.clear()
        for y, row in enumerate(self.ascii_rows):
            for x, c in enumerate(row):
                if c == "*":
                    self._chest_positions.add((x, y))
                elif c == "+":
                    self._stone_positions.add((x, y))
                elif c in ("X", "G"):
                    self._machine_positions.add((x, y))
                elif c == self.hero:
                    self._hero_x, self._hero_y = x, y

    def _encode_dynamic_grid(self):
        """Copie le static grid dans le buffer, puis ajoute canaux 9 (hero),
        10 (facing), 13-14 (look-ahead) selon l'etat courant."""
        # Copy static -> buffer (une seule operation vectorisee).
        np.copyto(self._grid_buf, self._static_grid)

        # Canal 9 : position du hero (one-hot).
        if 0 <= self._hero_x < self.width and 0 <= self._hero_y < self.height:
            self._grid_buf[self._HERO_CHANNEL, self._hero_y, self._hero_x] = 1.0

        # Canal 10 : facing (case devant le hero) - utile pour le lookahead
        # conditionnel cote Python (mirror HasChestAhead cote Godot).
        fx = self._hero_x + self._hero_facing[0]
        fy = self._hero_y + self._hero_facing[1]
        if 0 <= fx < self.width and 0 <= fy < self.height:
            self._grid_buf[self._HERO_FACING_CHANNEL, fy, fx] = 1.0

        # Canaux 13-14 : look-ahead machines. En squelette, on met les
        # positions futures = positions courantes des machines (anticipation
        # 0 tick). Un vrai moteur calculerait SimulateMachinePreview comme
        # cote Godot. Comme le hero ne bouge pas de toute facon quand il
        # n'y a pas de coffre devant, on garde cette approximation simple.
        # Si coffre devant (Twist rule) :
        if (fx, fy) in self._chest_positions:
            for (mx, my) in self._machine_positions:
                if 0 <= mx < self.width and 0 <= my < self.height:
                    # Simplifie : machines supposes ne pas bouger dans le
                    # squelette. Vrai environnement -> SimulateMachinePreview.
                    ch = self._NEXT_X_CHANNEL if self.machine_target == "X" else self._NEXT_G_CHANNEL
                    self._grid_buf[ch, my, mx] = 1.0

    # ------------------------------------------------------------------
    # Contrat du masque d'actions : mirror cote Godot (BuildValidActionMask).
    # L'environnement concret pourra surcharger cette methode pour ajouter
    # des contraintes specifiques (ex : HACK valide seulement si une machine
    # est sur la case), mais le contrat de base DOIT etre respecte.
    # ------------------------------------------------------------------
    def action_mask(self) -> np.ndarray:
        # On rempli self._mask_buf (pre-alloue) plutot que d'en creer un
        # nouveau a chaque appel : evite ~100 bytes/call de GC pressure.
        np.copyto(self._mask_buf, action_mask(self._is_on_engine, self.action_names))
        return self._mask_buf

    def reset(self, **kwargs):
        # Reset dynamique : positions initiales, ressources.
        self._scan_dynamic_entities()  # re-scanne au cas ou le precedent episode a mute les sets
        self._is_on_engine = False  # demarre hors engine
        self._battery = self._MAX_BATTERY
        self._stamina = self._MAX_STAMINA
        self._tick = 0
        self._hack_action_count = 0
        self._stones_collected = 0
        self._chests_hidden = 0
        self._chests_destroyed = 0
        self._invalid_streak = 0
        self._hero_facing = (1, 0)  # RIGHT par defaut

        # Encode l'observation initiale.
        self._encode_dynamic_grid()
        self._scalars_buf[:] = 0.0
        self._scalars_buf[0] = 1.0  # stamina = 100%
        self._scalars_buf[1] = 1.0  # battery = 100%
        self._scalars_buf[2] = 1.0  # temps = 100%
        self._scalars_buf[3] = self._hero_x / max(1, self.width - 1)
        self._scalars_buf[4] = self._hero_y / max(1, self.height - 1)
        self._scalars_buf[5] = 0.0  # is_on_engine = False au reset

        info = {
            "hero": self.hero,
            "hero_name": self.hero_name,
            "n_actions": self.n_actions,
            "is_on_engine": self._is_on_engine,
            "action_mask": self.action_mask().copy(),
            "hero_pos": (self._hero_x, self._hero_y),
        }
        return {"grid": self._grid_buf.copy(), "scalars": self._scalars_buf.copy()}, info

    def _terrain_at(self, x: int, y: int) -> str:
        """Retourne le caractere ASCII a (x,y) ou '#' si hors carte."""
        if y < 0 or y >= self.height or x < 0 or x >= self.width:
            return "#"
        return self.ascii_rows[y][x]

    def _elevation_at(self, x: int, y: int) -> int:
        if y < 0 or y >= self.height or x < 0 or x >= self.width:
            return 0
        return int(self.elevation[y, x])

    def _try_move(self, dx: int, dy: int) -> tuple[float, str]:
        """Tente un mouvement et renvoie (reward, resultat).
        Le reward direct est limite aux evenements du score. La progression vers
        les objectifs est calculee separement dans step().
        """
        tx, ty = self._hero_x + dx, self._hero_y + dy
        terrain = self._terrain_at(tx, ty)
        if self.hero == "F" and terrain in ("#", "t"):
            return 0.0, "blocked_move"
        if self.hero == "M" and terrain in ("#", "o"):
            return 0.0, "blocked_move"
        start_x, start_y = self._hero_x, self._hero_y
        hop = False
        if self.hero == "F" and terrain == "o":
            tx += dx
            ty += dy
            hop = True
        elif self.hero == "M" and terrain == "t":
            tx += dx
            ty += dy
            hop = True
        landing = self._terrain_at(tx, ty)
        if landing == "#":
            return 0.0, "blocked_move"
        if self.hero == "F" and landing in ("t", "o"):
            return 0.0, "blocked_move"
        if self.hero == "M" and landing in ("o", "t"):
            return 0.0, "blocked_move"
        start_elev = self._elevation_at(start_x, start_y)
        target_elev = self._elevation_at(tx, ty)
        delta = target_elev - start_elev
        if delta >= 3 or delta <= -5:
            return 0.0, "blocked_move"
        if hop:
            cost = 10.0
        elif delta > 0:
            cost = 3.0
        elif delta < 0:
            cost = 0.0
        else:
            cost = 1.0
        if self._stamina < cost:
            return 0.0, "insufficient_stamina"
        self._hero_x, self._hero_y = tx, ty
        self._stamina -= cost
        reward = 0.0
        result = "move"
        if (tx, ty) in self._stone_positions:
            self._stone_positions.discard((tx, ty))
            self._stones_collected += 1
            reward += float(self.reward_config.get("stone_collected", 25.0))
            result = "stone_collected"
        return reward, result

    def _try_push(self, dx: int, dy: int) -> tuple[float, str]:
        """Tente une poussée et renvoie (reward, résultat)."""
        rc = self.reward_config
        cx, cy = self._hero_x + dx, self._hero_y + dy
        if (cx, cy) not in self._chest_positions:
            return float(rc.get("push_without_chest", -0.35)), "push_without_chest"
        bx, by = cx + dx, cy + dy
        terrain_beyond = self._terrain_at(bx, by)
        if (
            terrain_beyond in ("#", "t")
            or (bx, by) in self._machine_positions
            or (bx, by) in self._chest_positions
        ):
            return float(rc.get("blocked_push", -0.25)), "blocked_push"
        chest_elev = self._elevation_at(cx, cy)
        target_elev = self._elevation_at(bx, by)
        # Les héros ne peuvent pas pousser un coffre en montée.
        if target_elev > chest_elev:
            return float(rc.get("blocked_push", -0.25)), "blocked_push"
        if self._stamina < self._PUSH_STAMINA_COST:
            return float(rc.get("blocked_push", -0.25)), "insufficient_stamina"
        old_dist = self._nearest_bush_distance(cx, cy)
        self._chest_positions.discard((cx, cy))
        self._stamina -= self._PUSH_STAMINA_COST
        self._hero_x, self._hero_y = cx, cy
        drop = chest_elev - target_elev
        if drop >= 5 or terrain_beyond == "o":
            self._chests_destroyed += 1
            return float(rc.get("voluntary_chest_loss", -50.0)), "chest_destroyed"
        if terrain_beyond == "@":
            self._chests_hidden += 1
            return float(rc.get("chest_hidden", 150.0)), "chest_hidden"
        self._chest_positions.add((bx, by))
        new_dist = self._nearest_bush_distance(bx, by)
        if new_dist < old_dist:
            return float(rc.get("progress_chest_to_bush", 0.75)), "chest_progress"
        if new_dist > old_dist:
            return float(rc.get("regress_chest_from_bush", -0.75)), "chest_regress"
        return 0.0, "chest_pushed"

    def _nearest_distance(
        self,
        x: int,
        y: int,
        targets: set[tuple[int, int]],
    ) -> int | None:
        if not targets:
            return None
        return min(abs(tx - x) + abs(ty - y) for tx, ty in targets)
    def _nearest_bush_distance(self, x: int, y: int) -> int:
        bushes = {
            (bx, by)
            for by, row in enumerate(self.ascii_rows)
            for bx, tile in enumerate(row)
            if tile == "@"
        }
        if not bushes:
            return self.width + self.height
        return min(abs(bx - x) + abs(by - y) for bx, by in bushes)
    def _objective_shaping(
        self,
        old_x: int,
        old_y: int,
        new_x: int,
        new_y: int,
    ) -> float:
        rc = self.reward_config
        reward = 0.0
        old_stone = self._nearest_distance(old_x, old_y, self._stone_positions)
        new_stone = self._nearest_distance(new_x, new_y, self._stone_positions)
        if old_stone is not None and new_stone is not None:
            if new_stone < old_stone:
                reward += float(rc.get("progress_to_stone", 0.30))
            elif new_stone > old_stone:
                reward += float(rc.get("regress_from_stone", -0.30))
        old_chest = self._nearest_distance(old_x, old_y, self._chest_positions)
        new_chest = self._nearest_distance(new_x, new_y, self._chest_positions)
        if old_chest is not None and new_chest is not None:
            if new_chest < old_chest:
                reward += float(rc.get("progress_to_chest", 0.40))
            elif new_chest > old_chest:
                reward += float(rc.get("regress_from_chest", -0.40))
        return reward

    def step(self, action):
        a = int(action)
        rc = self.reward_config
        if not 0 <= a < self.n_actions:
            action_name = "WAIT"
            is_invalid = True
        else:
            action_name = self.action_names[a]
            mask = self.action_mask()
            is_invalid = not bool(mask[a])
        reward = 0.0
        terminated = False
        truncated = False
        result = "invalid"
        old_x, old_y = self._hero_x, self._hero_y
        old_stamina = self._stamina
        old_battery = self._battery
        if is_invalid:
            self._invalid_streak += 1
            if action_name.startswith("PUSH_"):
                reward += float(rc.get("push_without_chest", -0.35))
                result = "invalid_push"
            elif action_name == "HACK":
                reward += float(rc.get("hack_without_machine", -0.35))
                result = "invalid_hack"
            elif action_name.startswith("HACK_"):
                reward += float(rc.get("hack_action_without_riding", -0.35))
                result = "invalid_hack_action"
            else:
                reward += float(rc.get("invalid_action", -0.25))
                result = "invalid"
            if self._invalid_streak >= 3:
                reward += float(rc.get("repeated_invalid_action", -0.50))
        else:
            self._invalid_streak = 0
            dir_map = {
                "UP": (0, -1),
                "DOWN": (0, 1),
                "LEFT": (-1, 0),
                "RIGHT": (1, 0),
            }
            push_dir_map = {
                "PUSH_UP": (0, -1),
                "PUSH_DOWN": (0, 1),
                "PUSH_LEFT": (-1, 0),
                "PUSH_RIGHT": (1, 0),
            }
            if action_name == "HACK":
                # Le squelette ne suit pas le type et la position de chaque
                # machine. Un HACK n'est accepte que si une machine est presente.
                if (self._hero_x, self._hero_y) not in self._machine_positions:
                    reward += float(rc.get("hack_without_machine", -0.35))
                    result = "hack_without_machine"
                elif self._battery < 1.0:
                    reward += float(rc.get("wasted_battery", -0.30))
                    result = "battery_empty"
                else:
                    self._is_on_engine = True
                    self._battery -= 1.0
                    self._hack_action_count = 0
                    result = "hack"
            elif action_name == "WAIT":
                if self._is_on_engine:
                    self._is_on_engine = False
                    self._hack_action_count = 0
                    result = "unhack"
                else:
                    if self._stamina <= 0.0 or self._battery <= 0.0:
                        reward += float(rc.get("wait_with_full_resources", -0.20))
                    else:
                        self._stamina = min(self._MAX_STAMINA, self._stamina + 0.5)
                    result = "wait"
            elif action_name in ("HACK_MOVE", "HACK_FILL", "HACK_CW", "HACK_CCW"):
                if self._battery < 1.0:
                    reward += float(rc.get("wasted_battery", -0.30))
                    result = "battery_empty"
                else:
                    self._battery -= 1.0
                    self._hack_action_count += 1
                    if action_name in ("HACK_CW", "HACK_CCW"):
                        reward += float(rc.get("useless_hack_rotation", -0.10))
                        result = "hack_rotation"
                    else:
                        reward += float(rc.get("useful_hack", 0.20))
                        result = action_name.lower()
            elif action_name in dir_map:
                self._hero_facing = dir_map[action_name]
                move_reward, result = self._try_move(*dir_map[action_name])
                reward += move_reward
                self._hack_action_count = 0
                if result == "move":
                    reward += self._objective_shaping(
                        old_x, old_y, self._hero_x, self._hero_y
                    )
                elif result in ("blocked_move", "insufficient_stamina"):
                    reward += float(rc.get("invalid_action", -0.25))
            elif action_name in push_dir_map:
                self._hero_facing = push_dir_map[action_name]
                push_reward, result = self._try_push(*push_dir_map[action_name])
                reward += push_reward
                self._hack_action_count = 0
        self._tick += 1
        if self._tick >= self.max_time:
            truncated = True
            reward += float(rc.get("timeout", 0.0))
        if self._stamina <= 0.0 and self._battery <= 0.0:
            terminated = True
            reward += float(rc.get("resource_exhausted", -5.0))
        self._encode_dynamic_grid()
        self._scalars_buf[0] = self._stamina / self._MAX_STAMINA
        self._scalars_buf[1] = self._battery / self._MAX_BATTERY
        self._scalars_buf[2] = max(
            0.0, 1.0 - self._tick / max(1, self.max_time)
        )
        self._scalars_buf[3] = self._hero_x / max(1, self.width - 1)
        self._scalars_buf[4] = self._hero_y / max(1, self.height - 1)
        self._scalars_buf[5] = 1.0 if self._is_on_engine else 0.0
        info = {
            "hero": self.hero,
            "action_name": action_name,
            "action_result": result,
            "is_on_engine": self._is_on_engine,
            "action_mask": self.action_mask().copy(),
            "invalid_action": is_invalid,
            "hero_pos": (self._hero_x, self._hero_y),
            "hero_stamina": self._stamina,
            "hero_battery": self._battery,
            "stamina_spent": max(0.0, old_stamina - self._stamina),
            "battery_spent": max(0.0, old_battery - self._battery),
            "tick": self._tick,
            "stones_collected": self._stones_collected,
            "chests_hidden": self._chests_hidden,
            "chests_destroyed": self._chests_destroyed,
            "termination_reason": (
                "resources_exhausted"
                if terminated
                else "timeout"
                if truncated
                else None
            ),
        }
        return (
            {
                "grid": self._grid_buf.copy(),
                "scalars": self._scalars_buf.copy(),
            },
            float(reward),
            bool(terminated),
            bool(truncated),
            info,
        )


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


class MapLoggerCallback(BaseCallback):
    """Log l'evolution de la map pour chaque session d'entrainement.

    Genere un fichier .log par session, contenant :
      - En-tete : date, hero, map source, elevation source, action list
      - Pour chaque episode logge (tous les --log-episodes) :
          * Episode N (step K)
          * Reward total, longueur, actions invalides
          * Etat final : stamina, battery, is_on_engine, hero_pos
          * Carte ASCII finale avec position du hero marquee (H)
          * Compteurs : pierres collectees, coffres caches, coffres detruits
      - Pour les premiers episodes de la session, log aussi les
        20 premieres actions pour aider a debugger les boucles (ex : LEFT forever)

    Le fichier est ecrit dans --log-dir (default: TrainingLogs/) sous le nom
    session_<hero>_<YYYYMMDD_HHMMSS>.log. Il est flushe apres chaque episode
    logge pour etre consultable en temps reel pendant l'entrainement.
    """

    def __init__(self, hero: str, log_dir: str, log_every: int = 50,
                 trace_first_episodes: int = 3, trace_actions_count: int = 20):
        super().__init__()
        self.hero = hero
        self.log_dir = Path(log_dir)
        self.log_every = max(1, log_every)
        self.trace_first_episodes = trace_first_episodes
        self.trace_actions_count = trace_actions_count
        self.episode_count = 0
        self._file = None
        self._session_start = None
        self._trace_buffer: list[str] = []  # actions du current episode
        self._trace_episode_idx = -1

    def _on_training_start(self) -> None:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = self.log_dir / f"session_{self.hero}_{session_id}.log"
        self._session_start = datetime.now()
        self._file = open(self.log_path, "w", encoding="utf-8")

        # En-tete : infos de session.
        self._file.write(f"# AlgoGames2 - session d'entrainement\n")
        self._file.write(f"# Hero: {self.hero} ({HEROES[self.hero]['name']})\n")
        self._file.write(f"# Date de debut: {self._session_start.isoformat()}\n")
        self._file.write(f"# N_actions: {HEROES[self.hero]['n_actions']}\n")
        self._file.write(f"# Action list: {HEROES[self.hero]['actions']}\n")
        # Extraire map_path et elevation_path depuis le premier env.
        try:
            env0 = self.model.env.envs[0].unwrapped
            self._file.write(f"# Map: {getattr(env0, 'map_path', '?')}\n")
            self._file.write(f"# Elevation: {getattr(env0, 'elevation_path', '?')}\n")
            self._file.write(f"# Map dimensions: {getattr(env0, 'height', '?')}x{getattr(env0, 'width', '?')}\n")
            self._file.write(f"# Max time: {getattr(env0, 'max_time', '?')} ticks\n")
            # Inclure la map ASCII initiale.
            if hasattr(env0, 'ascii_rows'):
                self._file.write(f"# Carte ASCII initiale:\n")
                for row in env0.ascii_rows:
                    self._file.write(f"#   {row}\n")
        except Exception as e:
            self._file.write(f"# (env introspection failed: {e})\n")
        self._file.write(f"\n")
        self._file.flush()
        print(f"[MapLogger] logging session to: {self.log_path}")

    def _on_step(self) -> bool:
        for i, info in enumerate(self.locals.get("infos", [])):
            # Tracer les N premieres actions des premiers episodes.
            action_name = info.get("action_name", "?")
            invalid = info.get("invalid_action", False)
            marker = "!" if invalid else " "
            # Si on est dans un episode a tracer, accumuler.
            episode = info.get("episode")
            if episode is not None:
                # Episode vient de finir, on va le logger ci-dessous si besoin.
                pass
            # Trace les actions de l'episode courant.
            if self._trace_episode_idx < self.trace_first_episodes:
                self._trace_buffer.append(f"{action_name}{marker}")

            # Si episode termine, on log le snapshot.
            if episode is not None:
                self.episode_count += 1
                self._trace_episode_idx += 1
                if self.episode_count % self.log_every == 0 or \
                   self._trace_episode_idx < self.trace_first_episodes:
                    self._log_episode(episode, info)
                # Reset le trace buffer pour le prochain episode.
                self._trace_buffer = []
        return True

    def _log_episode(self, episode, info):
        """Ecrit le snapshot d'un episode dans le fichier de log."""
        # SB3 Monitor wrapper fournit info["episode"] = {"r": reward, "l": length}
        # quand un episode se termine. On essaie les deux conventions.
        try:
            if isinstance(episode, dict):
                ep_reward = float(episode.get("r", 0.0))
                ep_length = int(episode.get("l", 0))
            else:
                ep_reward = float(getattr(episode, "r", 0.0) or getattr(episode, "episode_rewards", [0])[-1] if getattr(episode, "episode_rewards", None) else 0.0)
                ep_length = int(getattr(episode, "l", 0) or getattr(episode, "episode_lengths", [0])[-1] if getattr(episode, "episode_lengths", None) else 0)
        except Exception:
            ep_reward = 0.0
            ep_length = 0

        self._file.write(f"--- Episode {self.episode_count} (step {self.num_timesteps}) ---\n")
        self._file.write(f"  reward={ep_reward:.3f}  length={ep_length}\n")
        self._file.write(f"  is_on_engine={info.get('is_on_engine', False)}\n")
        self._file.write(f"  hero_pos={info.get('hero_pos', '?')}\n")
        self._file.write(f"  hero_stamina={info.get('hero_stamina', '?')}\n")
        self._file.write(f"  hero_battery={info.get('hero_battery', '?')}\n")
        self._file.write(f"  stones_collected={info.get('stones_collected', 0)}\n")
        self._file.write(f"  chests_hidden={info.get('chests_hidden', 0)}\n")
        self._file.write(f"  chests_destroyed={info.get('chests_destroyed', 0)}\n")

        # Carte ASCII avec position du hero marquee.
        try:
            env = self.model.env.envs[0].unwrapped
            if hasattr(env, 'ascii_rows'):
                hero_pos = info.get('hero_pos', None)
                hero_ch = "F" if self.hero == "F" else "M"
                # Si hero est sur engine, marquer avec minuscule (f/m).
                if info.get('is_on_engine', False):
                    hero_ch = hero_ch.lower()
                self._file.write(f"  map:\n")
                for y, row in enumerate(env.ascii_rows):
                    line_chars = list(row)
                    # Remplacer le hero original par '.' (le hero est encode dynamiquement)
                    for x, c in enumerate(line_chars):
                        if c == self.hero:
                            line_chars[x] = "."
                    # Marquer la position courante du hero.
                    if hero_pos is not None and hero_pos[1] == y:
                        # Bounds check
                        if 0 <= hero_pos[0] < len(line_chars):
                            line_chars[hero_pos[0]] = hero_ch
                    self._file.write(f"    {''.join(line_chars)}\n")
        except Exception as e:
            self._file.write(f"  (map snapshot failed: {e})\n")

        # Trace des premieres actions (pour debug des boucles).
        if self._trace_episode_idx < self.trace_first_episodes and self._trace_buffer:
            self._file.write(f"  first_actions (max {self.trace_actions_count}): ")
            self._file.write(" ".join(self._trace_buffer[:self.trace_actions_count]))
            if len(self._trace_buffer) > self.trace_actions_count:
                self._file.write(f" ... ({len(self._trace_buffer)} total)")
            self._file.write("\n")

        self._file.write("\n")
        self._file.flush()

    def _on_training_end(self) -> None:
        if self._file:
            duration = datetime.now() - self._session_start
            self._file.write(f"\n# Training ended: {datetime.now().isoformat()}\n")
            self._file.write(f"# Duration: {duration}\n")
            self._file.write(f"# Total episodes: {self.episode_count}\n")
            self._file.close()
            print(f"[MapLogger] session log saved: {self.log_path}")


def choose_augmentation(mode: str, rank: int) -> str:
    """Conserve pour compatibilite mais N'EST PLUS appele par make_single_env
    (cf. diagnostic Cause 3) : resoudre "random"/"all" une seule fois ici,
    a la creation de l'env, figeait la meme augmentation pour tout
    l'entrainement -> desaccord possible avec l'evaluation Optuna, qui elle
    force toujours "identity". AlgoEnv.reset() retire desormais une nouvelle
    augmentation a CHAQUE episode via AlgoEnv._pick_augmentation()."""
    if mode == "all":
        return AUGMENTATIONS[rank % len(AUGMENTATIONS)]
    if mode == "random":
        return random.choice(AUGMENTATIONS)
    return mode if mode in AUGMENTATIONS else "identity"


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
            # Mode brut ("identity"/"random"/"all"/nom fixe), non resolu ici :
            # AlgoEnv retire une augmentation a chaque reset() (Cause 3).
            "augmentation": augmentation,
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


def build_policy_kwargs(n_actions: int) -> dict:                         # MIGRATION MASKABLEPPO
    """Construit les policy_kwargs pour une politique feedforward.

    MaskablePPO utilise une `ActorCriticPolicy` feedforward. On retire
    toute configuration LSTM et on garde un `net_arch` léger.
    """
    return {
        "features_extractor_class": GridScalarExtractor,
        "features_extractor_kwargs": {"features_dim": 128},
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

    # Charge AlgoEnv.py (moteur GDD+Twist reel) s'il est present a cote de ce
    # script ; sinon on retombe sur l'environnement embarque (squelette).
    algo_env_path = Path(__file__).resolve().parent / "AlgoEnv.py"
    if algo_env_path.is_file():
        external_module = import_module(algo_env_path, "algogames_env")
        factory = find_factory(external_module)
        print(f"[train] moteur externe charge depuis {algo_env_path.name} -> {factory.__name__}")
    else:
        factory = AlgoGamesEnv
        print("[train] AlgoEnv.py introuvable : utilisation du moteur embarque (squelette).")

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

    # EvalCallback desactive si --no-eval (recommande sur CPU ; l'eval double
    # la charge CPU pendant les eval episodes sans grand gain sur de petits
    # modeles).
    eval_env = None
    if not args.no_eval:
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

    # MaskablePPO feedforward policy kwargs (lstm removed).            # MIGRATION MASKABLEPPO
    policy_kwargs = build_policy_kwargs(n_actions)

    rollout_size = args.n_steps * args.n_envs
    batch_size = min(args.batch_size, rollout_size)
    while rollout_size % batch_size != 0 and batch_size > 1:
        batch_size -= 1

    if args.resume:
        resume_path = resolve_file(args.resume, "Modele a reprendre")
        model = MaskablePPO.load(                                            # MIGRATION MASKABLEPPO
            str(resume_path),
            env=train_env,
            device=args.device,
        )
    else:
        best_params_path = hyperparams_path(hero)
        best_params: dict[str, Any] = {}
        if best_params_path.exists():
            with best_params_path.open("r", encoding="utf-8") as f:
                best_params = json.load(f)
            print(f"[train] hyperparametres Optuna charges depuis {best_params_path}: {best_params}")
        else:
            print(f"[train] aucun fichier Optuna trouve ({best_params_path.name}); utilisation des parametres par defaut.")

        model_kwargs: dict[str, Any] = {
            "policy": "MultiInputPolicy",                                 # MIGRATION MASKABLEPPO
            "env": train_env,
            "device": args.device,
            "policy_kwargs": policy_kwargs,
            "verbose": 1,
            "seed": args.seed,
            "learning_rate": args.learning_rate,
            "n_steps": args.n_steps,
            "batch_size": batch_size,
            "gamma": args.gamma,
            "gae_lambda": args.gae_lambda,
            "ent_coef": args.ent_coef,
            "tensorboard_log": str(output_path.parent / f"tensorboard_{hero}"),
        }
        model_kwargs.update(best_params)
        model = MaskablePPO(**model_kwargs)                                # MIGRATION MASKABLEPPO

    checkpoint_dir = output_path.parent / f"checkpoints_{hero}"
    best_dir = output_path.parent / f"best_{hero}"
    checkpoint_dir.mkdir(exist_ok=True)
    best_dir.mkdir(exist_ok=True)

    # Callbacks : stats + map logger + checkpoint + (optionnel) eval.
    callbacks: list[BaseCallback] = [
        EpisodeStatsCallback(hero=hero),
        MapLoggerCallback(
            hero=hero,
            log_dir=args.log_dir,
            log_every=args.log_episodes,
        ),
    ]

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

    if not args.no_eval and eval_env is not None and args.eval_freq > 0:
        callbacks.append(
            MaskableEvalCallback(                                       # MIGRATION MASKABLEPPO
                eval_env,
                best_model_save_path=str(best_dir),
                log_path=str(best_dir),
                eval_freq=max(args.eval_freq // args.n_envs, 1),
                n_eval_episodes=5,
                deterministic=True,
                render=False,
                use_masking=True,                                       # MIGRATION MASKABLEPPO
            )
        )

    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=CallbackList(callbacks),
            reset_num_timesteps=not bool(args.resume),
            progress_bar=not args.no_progress_bar,
        )
        model.save(str(output_path.with_suffix("")))
        print(f"[train] modele sauvegarde : {output_path}")
        print(
            f"[train] hero={HEROES[hero]['name']} carte={height}x{width} "
            f"temps={max_time} "
            f"observation=({N_GRID_CHANNELS}, {MAX_HEIGHT}, {MAX_WIDTH})+"
            f"{N_SCALARS} actions={n_actions} "
            f"policy=feedforward device={args.device}"
        )
        print(
            f"[train] actions : {HEROES[hero]['actions']}"
        )
    finally:
        train_env.close()
        if eval_env is not None:
            eval_env.close()


if __name__ == "__main__":
    main()
