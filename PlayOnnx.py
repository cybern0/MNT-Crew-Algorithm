"""PlayOnnx.py — rejoue une partie avec les deux ONNX et ecrit actions.txt.

    python PlayOnnx.py --map map.txt --elevation elevation.txt \
        --onnx-f OnnxModels/ikotofosa.onnx --onnx-m OnnxModels/imahaki.onnx

Meme contrat d'inference que le runtime C# (logits + argmax masque cote
appelant) et MEME masque qu'a l'entrainement : legal_action_mask, moins WAIT
apres WAIT_STREAK_LIMIT WAIT consecutifs. Aucune heuristique aleatoire n'est
necessaire : la boucle WAIT infinie est impossible par construction du masque.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

from AlgoSpec import (
    HEROES, RESOURCE_LOW_TICKS_LIMIT, WAIT_STREAK_LIMIT,
    format_actions_lines, load_map, resolve_file,
)
from AlgoEnv import build_grid, build_scalars
from GameEngine import GameEngine


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rejoue AlgoGames 2 avec les ONNX F et M.")
    p.add_argument("--map", required=True, dest="map_path")
    p.add_argument("--elevation", required=True, dest="elevation_path")
    p.add_argument("--actions", default="actions.txt")
    p.add_argument("--onnx-f", default="OnnxModels/ikotofosa.onnx")
    p.add_argument("--onnx-m", default="OnnxModels/imahaki.onnx")
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--tick-delay", type=float, default=0.0)
    p.add_argument("--render", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_session(path: str, device: str) -> ort.InferenceSession:
    providers = (["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "cuda"
                 else ["CPUExecutionProvider"])
    return ort.InferenceSession(str(resolve_file(path, "modele ONNX")), providers=providers)


def mask_for(engine: GameEngine, hero: str, names: list[str], wait_streak: int) -> np.ndarray:
    mask = engine.legal_action_mask(hero, names).copy()
    wait = names.index("WAIT")
    if wait_streak >= WAIT_STREAK_LIMIT and mask[wait] and mask.sum() > 1:
        mask[wait] = False
    return mask


def pick_action(session, names, grid, scalars, mask) -> str:
    logits = session.run(None, {"map": grid[None].astype(np.float32),
                                "stats": scalars[None].astype(np.float32)})[0][0]
    masked = np.where(mask, logits, -np.inf)
    if not np.isfinite(masked).any():
        return "WAIT"
    return names[int(np.argmax(masked))]


def render(engine: GameEngine) -> str:
    grid = [row[:] for row in engine.terrain]
    for sx, sy in engine.stones:
        grid[sy][sx] = "+"
    for cx, cy in engine.chests:
        grid[cy][cx] = "*"
    for m in engine.machines:
        grid[m["y"]][m["x"]] = m["type"]
    for h in ("F", "M"):
        hx, hy = engine.pos[h]
        grid[hy][hx] = h
    return "\n".join("".join(row) for row in grid)


def main() -> int:
    args = parse_args()
    spec = load_map(args.map_path, args.elevation_path)
    engine = GameEngine(list(spec.rows), spec.elevation, spec.max_time, seed=args.seed)
    sessions = {"F": load_session(args.onnx_f, args.device),
                "M": load_session(args.onnx_m, args.device)}
    names = {h: HEROES[h]["actions"] for h in ("F", "M")}

    # actions.txt doit tenir en max_time lignes, END_GAME incluse.
    max_action_ticks = max(0, spec.max_time - 1)
    pairs: list[tuple[str, str]] = []
    streak = {"F": 0, "M": 0}

    print(f"[play] carte {spec.name}, temps limite={spec.max_time}")
    tick = 0
    while tick < max_action_ticks:
        chosen = {}
        for h in ("F", "M"):
            mask = mask_for(engine, h, names[h], streak[h])
            chosen[h] = pick_action(
                sessions[h], names[h],
                build_grid(engine, h, spec.elevation),
                build_scalars(engine, h), mask)
        ev = engine.step(chosen)
        for h in ("F", "M"):
            idle = chosen[h] == "WAIT" or ev[h]["kind"] in ("wait", "implicit_wait")
            streak[h] = streak[h] + 1 if idle else 0

        tick += 1
        pairs.append((chosen["F"], chosen["M"]))

        if args.render:
            print(f"\n=== tick {tick} ===\n{render(engine)}")
        print(f"[tick {tick}] F={chosen['F']:<11} M={chosen['M']:<11} "
              f"stones={engine.stones_collected} hidden={engine.chests_hidden} "
              f"score={engine.official_score()}")

        if args.tick_delay > 0:
            time.sleep(args.tick_delay)

        cleared = not engine.stones and not engine.chests
        exhausted = any(engine.resource_low_ticks[h] >= RESOURCE_LOW_TICKS_LIMIT
                        for h in ("F", "M"))
        if cleared or exhausted:
            print(f"[play] fin au tick {tick} "
                  f"({'objectifs atteints' if cleared else 'ressources epuisees'}).")
            break

    path = Path(args.actions)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(format_actions_lines(pairs)) + "\n", encoding="utf-8")
    print(f"[play] {tick} ticks, actions -> {path}")
    print(f"[play] score final officiel = {engine.official_score()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
