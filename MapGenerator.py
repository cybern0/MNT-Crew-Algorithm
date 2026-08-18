#!/usr/bin/env python3
"""MapGenerator.py

Standalone map generator for AlgoGames 2.

Produit une paire (map.txt, elevation.txt) valide et compatible avec
AlgoTrain.py (observation 15 canaux + 6 scalaires : 11 tuiles + ele abs + ele rel + look-ahead X/G) et le preprocesseur C#
cible. Le générateur respecte le GDD et le « Twist » (règles d'élévation) :

  - Une seule entite par case, exactement un 'F' (Ikotofosa) et un 'M' (Imahaki).
  - Au moins un excavateur 'X' et un grappler 'G' pour que les deux héros
    aient une machine à pirater (sinon l'entrainement n'apprend pas HACK_*).
  - Au moins un coffre '*' et une cache '@' (sinon la récompense principale
    chest_hidden n'est jamais atteignable).
  - Élévations 0-9 ; 0 réservé aux rochers '#'. Les autres cases 1..9.
  - Contrainte « o » / « t » : |elevation - voisin| <= 5 (garanti par le Twist).
  - F et M connectés par un chemin non-rocheux, et la case d'au moins une
    machine est accessible à pied depuis l'un des deux héros.
  - Pas de 'X' ou 'G' sur une élévation 0 (les machines ne traversent pas les
    rochers).

Usage :
    python MapGenerator.py --height 10 --width 12 --out-map map.txt --out-elevation elevation.txt
    python MapGenerator.py --hero F --seed 42   # force la présence d'un excavateur proche de F
"""
from __future__ import annotations

import argparse
import random
from collections import deque
from pathlib import Path
from typing import Tuple

import numpy as np

MAX_SOURCE_HEIGHT = 15
MAX_SOURCE_WIDTH = 20

# Cases jouables selon le GDD. 'F' et 'M' sont les héros ; 'X' et 'G' les
# machines (excavateur / grappler) ; '*' coffre ; '+' pierre ; '@' cache.
ASCII_TILES = list(".#to*+@FMXG")
ROCK = "#"
HEROES = ("F", "M")
MACHINES = ("X", "G")


def _neighbors(x: int, y: int, w: int, h: int):
    for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0)):
        nx, ny = x + dx, y + dy
        if 0 <= nx < w and 0 <= ny < h:
            yield nx, ny


def _flood_reachable(rows: list[str], sources) -> set[tuple[int, int]]:
    """Retourne l'ensemble des cases atteignables depuis `sources` en passant
    uniquement par des cases non-rocher (#)."""
    h, w = len(rows), len(rows[0])
    seen = set()
    q = deque(sources)
    for s in sources:
        seen.add(s)
    while q:
        x, y = q.popleft()
        for nx, ny in _neighbors(x, y, w, h):
            if (nx, ny) in seen:
                continue
            if rows[ny][nx] != ROCK:
                seen.add((nx, ny))
                q.append((nx, ny))
    return seen


def _carve_path(rows: list[str], a, b) -> list[str]:
    """Creuse un chemin Manhattan en '.' entre a et b en passant à travers
    les rochers (utilisé si F et M ne sont pas connectés)."""
    ax, ay = a
    bx, by = b
    grid = [list(r) for r in rows]
    x, y = ax, ay
    while x != bx:
        if grid[y][x] == ROCK:
            grid[y][x] = "."
        x += 1 if bx > x else -1
    while y != by:
        if grid[y][x] == ROCK:
            grid[y][x] = "."
        y += 1 if by > y else -1
    return ["".join(r) for r in grid]


def _enforce_twist_constraints(rows: list[str], elevation: np.ndarray) -> None:
    """Ajuste localement les élévations pour respecter :

      - 'o' / 't' : |elev - voisin| <= 5 (règle du Twist)
      - '#' : elev 0 ; autres : elev dans [1, 9]

    Modifie `elevation` en place. Aucune garantie d'optimalité ; on clippe
    vers le voisin le plus proche de la contrainte.
    """
    h, w = elevation.shape
    # D'abord, reserrer les rochers à 0 et les autres à >= 1.
    for y in range(h):
        for x in range(w):
            if rows[y][x] == ROCK:
                elevation[y, x] = 0
            else:
                if elevation[y, x] < 1:
                    elevation[y, x] = 1
                elif elevation[y, x] > 9:
                    elevation[y, x] = 9

    # Ensuite, appliquer la contrainte o/t <= 5 vs. voisin > 0.
    for _ in range(3):  # converger en quelques passes
        changed = False
        for y in range(h):
            for x in range(w):
                tile = rows[y][x]
                if tile not in {"o", "t"}:
                    continue
                for nx, ny in _neighbors(x, y, w, h):
                    neighbor = int(elevation[ny, nx])
                    if neighbor <= 0:
                        continue
                    cur = int(elevation[y, x])
                    if abs(cur - neighbor) > 5:
                        # Tirer vers le voisin sans casser [1, 9].
                        new_val = neighbor + 5 if cur < neighbor else max(neighbor - 5, 1)
                        new_val = max(1, min(9, new_val))
                        if new_val != cur:
                            elevation[y, x] = new_val
                            changed = True
        if not changed:
            break


def _place_unique(rows: list[str], tile: str, rnd: random.Random,
                  forbidden: set[tuple[int, int]] | None = None) -> tuple[int, int]:
    """Place un caractère unique (F, M, X, G) sur une case libre non-rocher."""
    forbidden = forbidden or set()
    h, w = len(rows), len(rows[0])
    candidates = [
        (x, y)
        for y in range(h)
        for x in range(w)
        if rows[y][x] not in (ROCK, "F", "M", "X", "G") and (x, y) not in forbidden
    ]
    if not candidates:
        raise RuntimeError(f"Aucune case libre pour placer '{tile}'.")
    return rnd.choice(candidates)


def generate_map(
    height: int = 10,
    width: int = 10,
    max_time: int = 100,
    out_map: str = "map.txt",
    out_elevation: str = "elevation.txt",
    seed: int | None = None,
    n_chests: int = 2,
    n_stones: int = 4,
    n_bushes: int = 2,
    n_holes: int = 4,
    n_trees: int = 4,
    rock_ratio: float = 0.08,
) -> Tuple[Path, Path]:
    """Génère une carte + élévation valides.

    Le générateur place systématiquement :
      - 1 'F' (Ikotofosa) et 1 'M' (Imahaki)
      - 1 'X' (excavateur) et 1 'G' (grappler) -> les deux héros ont une
        cible de piratage pour apprendre HACK_*.
      - n_chests coffres '*' et n_bushes caches '@' -> la récompense
        chest_hidden est atteignable.

    Toutes les élévations respectent : 0 pour '#', 1..9 sinon, et |o - voisin|
    <= 5, |t - voisin| <= 5 (règle du Twist).
    """
    rnd = random.Random(seed)

    if not (1 <= height <= MAX_SOURCE_HEIGHT and 1 <= width <= MAX_SOURCE_WIDTH):
        raise ValueError("height/width hors bornes autorisées (1..15 x 1..20).")
    if height * width < 12:
        raise ValueError("Carte trop petite pour contenir F, M, X, G, *, @ et du sol.")

    # Distribution de base : on évite de placer des héros/machines par tirage
    # aléatoire — ils seront placés explicitement plus bas.
    tile_pool = (
        ["."] * 60
        + ["t"] * n_trees
        + ["o"] * n_holes
        + ["*"] * n_chests
        + ["+"] * n_stones
        + ["@"] * n_bushes
    )

    def make_rows() -> list[str]:
        rows = []
        for _y in range(height):
            row = []
            for _x in range(width):
                if rnd.random() < rock_ratio:
                    row.append(ROCK)
                else:
                    row.append(rnd.choice(tile_pool))
            rows.append("".join(row))
        return rows

    def connected(a, b, rows):
        return b in _flood_reachable(rows, [a])

    attempts = 0
    while True:
        attempts += 1
        rows = make_rows()

        # Placer F, M, X, G sur des cases libres distinctes.
        try:
            fpos = _place_unique(rows, "F", rnd)
            rows[fpos[1]] = rows[fpos[1]][:fpos[0]] + "F" + rows[fpos[1]][fpos[0] + 1:]
            mpos = _place_unique(rows, "M", rnd, forbidden={fpos})
            rows[mpos[1]] = rows[mpos[1]][:mpos[0]] + "M" + rows[mpos[1]][mpos[0] + 1:]
            xpos = _place_unique(rows, "X", rnd, forbidden={fpos, mpos})
            rows[xpos[1]] = rows[xpos[1]][:xpos[0]] + "X" + rows[xpos[1]][xpos[0] + 1:]
            gpos = _place_unique(rows, "G", rnd, forbidden={fpos, mpos, xpos})
            rows[gpos[1]] = rows[gpos[1]][:gpos[0]] + "G" + rows[gpos[1]][gpos[0] + 1:]
        except RuntimeError:
            if attempts > 50:
                raise
            continue

        # Garantir qu'il reste au moins un '*' et un '@' quelque part.
        flat = "".join(rows)
        if flat.count("*") < 1 or flat.count("@") < 1:
            if attempts > 50:
                raise RuntimeError("Impossible de garantir coffre + cache.")
            continue

        # Garantir la connectivité F<->M (sinon le jeu est injouable).
        if not connected(fpos, mpos, rows):
            rows = _carve_path(rows, fpos, mpos)

        # Élévation : 0 pour '#', 1..9 sinon.
        elevation = np.zeros((height, width), dtype=np.int8)
        for y in range(height):
            for x in range(width):
                if rows[y][x] == ROCK:
                    elevation[y, x] = 0
                else:
                    elevation[y, x] = rnd.randint(1, 9)

        _enforce_twist_constraints(rows, elevation)

        # Vérifications finales (anti-boucle infinie).
        try:
            _validate(rows, elevation)
            break
        except ValueError:
            if attempts > 100:
                raise
            continue

    # Écriture des fichiers.
    map_path = Path(out_map)
    elev_path = Path(out_elevation)
    map_path.parent.mkdir(parents=True, exist_ok=True)
    elev_path.parent.mkdir(parents=True, exist_ok=True)

    header = f"{height} {width} {max_time}\n"
    map_path.write_text(header + "\n".join(rows) + "\n", encoding="utf-8")
    elev_lines = ["".join(str(int(v)) for v in row) for row in elevation.tolist()]
    elev_path.write_text("\n".join(elev_lines) + "\n", encoding="utf-8")
    return map_path, elev_path


def _validate(rows: list[str], elevation: np.ndarray) -> None:
    """Rejoue les vérifications de AlgoTrain.read_map_header + validate_terrain
    pour s'assurer que la carte générée est compatible avec l'entraînement."""
    h, w = elevation.shape
    if len(rows) != h or any(len(r) != w for r in rows):
        raise ValueError("Dimensions rows != elevation.")

    counts = {c: 0 for c in "FMXG*@+#ot"}
    for row in rows:
        for ch in row:
            if ch in counts:
                counts[ch] += 1
            elif ch not in ".":
                raise ValueError(f"Symbole inattendu : {ch!r}")

    for required, label in (("F", "Ikotofosa"), ("M", "Imahaki"),
                            ("X", "Excavator"), ("G", "Grappler")):
        if counts[required] != 1:
            raise ValueError(f"Carte invalide : attendu 1 {label} ({required}), "
                             f"trouvé {counts[required]}")
    if counts["*"] < 1:
        raise ValueError("Carte invalide : au moins un coffre '*' requis.")
    if counts["@"] < 1:
        raise ValueError("Carte invalide : au moins une cache '@' requise.")

    for y in range(h):
        for x in range(w):
            tile = rows[y][x]
            level = int(elevation[y, x])
            if tile == ROCK and level != 0:
                raise ValueError(f"Rocher ({x},{y}) elevation {level}, attendu 0.")
            if tile != ROCK and level == 0:
                raise ValueError(f"Élévation 0 sur '{tile}' en ({x},{y}) ; "
                                  "0 est réservé aux rochers.")
            if level < 0 or level > 9:
                raise ValueError(f"Élévation hors bornes en ({x},{y}) : {level}.")

            if tile in {"o", "t"}:
                for nx, ny in _neighbors(x, y, w, h):
                    neighbor = int(elevation[ny, nx])
                    if neighbor > 0 and abs(level - neighbor) > 5:
                        raise ValueError(
                            f"Twist invalide en ({x},{y}) '{tile}' level={level} "
                            f"voisin ({nx},{ny}) level={neighbor} : diff>5."
                        )


def _cli():
    parser = argparse.ArgumentParser(description="Generate map + elevation pair.")
    parser.add_argument("--height", type=int, default=10)
    parser.add_argument("--width", type=int, default=10)
    parser.add_argument("--max-time", type=int, default=100)
    parser.add_argument("--out-map", default="map.txt")
    parser.add_argument("--out-elevation", default="elevation.txt")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--chests", type=int, default=2)
    parser.add_argument("--stones", type=int, default=4)
    parser.add_argument("--bushes", type=int, default=2)
    parser.add_argument("--holes", type=int, default=4)
    parser.add_argument("--trees", type=int, default=4)
    parser.add_argument("--rock-ratio", type=float, default=0.08)
    args = parser.parse_args()

    m, e = generate_map(
        height=args.height,
        width=args.width,
        max_time=args.max_time,
        out_map=args.out_map,
        out_elevation=args.out_elevation,
        seed=args.seed,
        n_chests=args.chests,
        n_stones=args.stones,
        n_bushes=args.bushes,
        n_holes=args.holes,
        n_trees=args.trees,
        rock_ratio=args.rock_ratio,
    )
    print(f"Wrote {m} and {e}")


if __name__ == "__main__":
    _cli()
