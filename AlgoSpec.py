"""AlgoSpec.py — contrat partage : constantes d'observation, catalogue d'actions,
lecture/validation des cartes, format du fichier d'actions.

Ne depend ni de gymnasium ni de torch : importable par les tests, l'export ONNX
et le script d'inference sans tirer la stack d'entrainement.
"""
from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import numpy as np

# --- Contrat d'observation (identique cote C#/ONNX) -------------------------
MAX_HEIGHT, MAX_WIDTH = 30, 30
MAX_SOURCE_HEIGHT, MAX_SOURCE_WIDTH = 15, 20
N_GRID_CHANNELS, N_SCALARS = 15, 6

TILE_CHANNELS = {".": 0, "#": 1, "*": 2, "o": 3, "t": 4, "@": 5, "+": 6, "X": 7, "G": 8}
HERO_CH, FACING_CH = 9, 10
ABS_ELEV_CH, REL_ELEV_CH = 11, 12
NEXT_X_CH, NEXT_G_CH = 13, 14

LOOKAHEAD_TICKS = 2
LOOKAHEAD_ONLY_IF_CHEST_AHEAD = True

ASCII_TILES = frozenset(".#to*+@FMXG")

# --- Catalogue d'actions (ordre CONTRACTUEL, mirroir du switch C#) ----------
IKOTOFOSA_ACTIONS = [
    "UP", "DOWN", "LEFT", "RIGHT", "WAIT",
    "PUSH_UP", "PUSH_DOWN", "PUSH_LEFT", "PUSH_RIGHT",
    "HACK", "HACK_MOVE", "HACK_FILL", "HACK_CW", "HACK_CCW",
]
IMAHAKI_ACTIONS = [
    "UP", "DOWN", "LEFT", "RIGHT", "WAIT",
    "PUSH_UP", "PUSH_DOWN", "PUSH_LEFT", "PUSH_RIGHT",
    "HACK", "HACK_MOVE", "HACK_CW", "HACK_CCW",
]

HEROES = {
    "F": {"name": "ikotofosa", "actions": IKOTOFOSA_ACTIONS,
          "n_actions": len(IKOTOFOSA_ACTIONS), "machine": "X"},
    "M": {"name": "imahaki", "actions": IMAHAKI_ACTIONS,
          "n_actions": len(IMAHAKI_ACTIONS), "machine": "G"},
}

# --- Recompense : DEUX termes derives du score, plus un bonus d'exploration -
# score_scale     : 1 pierre = +1.0 de reward.
# chest_destroyed : -150/25, cout d'opportunite exact d'un coffre perdu.
# novelty_bonus   : accorde UNE SEULE FOIS par case et par episode, donc
#                   strictement non-farmable (un retour sur une case connue
#                   rapporte exactement zero).
# Toute cle ajoutee ici est du shaping : test_reward_config_has_no_shaping_keys
# echouera. C'est voulu.
REWARD_CONFIG = {
    "score_scale": 25.0,
    "chest_destroyed": -6.0,
    "novelty_bonus": 0.05,
}

# --- Contraintes structurelles (pas de la recompense) ----------------------
WAIT_STREAK_LIMIT = 5        # apres 5 WAIT, WAIT sort du masque si alternative
RESOURCE_LOW_TICKS_LIMIT = 10  # fin d'episode : plus rien a depenser

# --- Reverse curriculum : distribution des etats de depart -----------------
# (lo, hi) = bande de distance de Manhattan a l'objectif le plus proche.
# "machine" = spawn sur la machine cible du hero. None = positions du GDD.
CURRICULUM_PHASES = ((1, 2), (3, 5), "machine", None)


class MapSpec(NamedTuple):
    """Une carte validee, prete a instancier un GameEngine."""
    name: str
    rows: tuple[str, ...]
    elevation: np.ndarray
    max_time: int


def resolve_file(path: str, label: str) -> Path:
    result = Path(path).expanduser().resolve()
    if not result.is_file():
        raise FileNotFoundError(f"{label} introuvable : {result}")
    return result


def read_map_header(path: Path) -> tuple[int, int, int, list[str]]:
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    if not lines:
        raise ValueError(f"{path} est vide.")
    header = lines[0].split()
    if len(header) != 3:
        raise ValueError("Premiere ligne attendue : H W temps.")
    height, width, max_time = (int(v) for v in header)
    if not 1 <= height <= MAX_SOURCE_HEIGHT:
        raise ValueError(f"H hors bornes 1..{MAX_SOURCE_HEIGHT}.")
    if not 1 <= width <= MAX_SOURCE_WIDTH:
        raise ValueError(f"W hors bornes 1..{MAX_SOURCE_WIDTH}.")
    if max_time <= 0:
        raise ValueError("temps doit etre > 0.")
    rows = lines[1:1 + height]
    if len(rows) != height or any(len(r) != width for r in rows):
        raise ValueError("Dimensions ASCII incoherentes avec H/W.")
    unknown = sorted({c for r in rows for c in r} - ASCII_TILES)
    if unknown:
        raise ValueError(f"Symboles inconnus : {unknown}")
    for hero in ("F", "M"):
        if sum(r.count(hero) for r in rows) != 1:
            raise ValueError(f"La carte doit contenir exactement un {hero}.")
    return height, width, max_time, rows


def read_elevation(path: Path, height: int, width: int) -> np.ndarray:
    rows: list[list[int]] = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        s = line.strip()
        if not s:
            continue
        parts = s.replace(",", " ").split()
        if len(parts) == 1 and len(parts[0]) == width and parts[0].isdigit():
            rows.append([int(c) for c in parts[0]])
        else:
            rows.append([int(v) for v in parts])
    if len(rows) != height or any(len(r) != width for r in rows):
        raise ValueError("Dimensions d'elevation incoherentes avec la carte.")
    elevation = np.asarray(rows, dtype=np.int8)
    if np.any((elevation < 0) | (elevation > 9)):
        raise ValueError("Elevations attendues entre 0 et 9.")
    # L'elevation ne doit JAMAIS muter pendant un episode (contrat du Twist) :
    # on la rend readonly, toute ecriture accidentelle leve immediatement.
    elevation.setflags(write=False)
    return elevation


def validate_terrain(rows: list[str], elevation: np.ndarray) -> None:
    """Elevation 0 = infranchissable, donc reservee aux rochers, et un rocher
    doit etre a 0. Les trous/arbres restent a moins de 5 niveaux d'un voisin."""
    height, width = elevation.shape
    for y, row in enumerate(rows):
        for x, tile in enumerate(row):
            level = int(elevation[y, x])
            if tile == "#" and level != 0:
                raise ValueError(f"Rocher en ({x},{y}) a l'elevation {level}, attendu 0.")
            if tile != "#" and level == 0:
                raise ValueError(f"Elevation 0 en ({x},{y}) sous '{tile}'.")
            if tile not in ("o", "t"):
                continue
            for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0)):
                nx, ny = x + dx, y + dy
                if 0 <= nx < width and 0 <= ny < height:
                    neighbor = int(elevation[ny, nx])
                    if neighbor > 0 and abs(level - neighbor) > 5:
                        raise ValueError(f"Twist invalide en ({x},{y}) pour '{tile}'.")


def load_map(map_path: str, elevation_path: str) -> MapSpec:
    mp = resolve_file(map_path, "map.txt")
    ep = resolve_file(elevation_path, "elevation.txt")
    height, width, max_time, rows = read_map_header(mp)
    elevation = read_elevation(ep, height, width)
    validate_terrain(rows, elevation)
    return MapSpec(mp.stem, tuple(rows), elevation, max_time)


def load_maps(map_paths: list[str], elevation_paths: list[str]) -> list[MapSpec]:
    """Variation reelle de topologie : plusieurs cartes appariees, tirees au
    sort a chaque reset(). Remplace les augmentations geometriques, qui
    preservaient la topologie et donc les memes pieges."""
    if len(map_paths) != len(elevation_paths):
        raise ValueError("--map et --elevation doivent avoir le meme nombre d'entrees.")
    return [load_map(m, e) for m, e in zip(map_paths, elevation_paths)]


def format_actions_lines(pairs: list[tuple[str, str]]) -> list[str]:
    """Format GDD : une ligne "ACTION_F | ACTION_M" par tick, END_GAME en
    derniere ligne (GameRunner honore un END_GAME anticipe sans penalite)."""
    return [f"{a_f} | {a_m}" for a_f, a_m in pairs] + ["END_GAME"]
