"""PlayOnnx.py — Rejoue AlgoGames 2 avec les 2 modeles ONNX exportes (F et M),
sans passer par stable-baselines3 : c'est le meme contrat d'inference que le
runtime C# (map [1,15,30,30] + stats [1,6] -> logits [1,n_actions], argmax
masque cote appelant, cf. Exports.py).

Log l'evolution de la map a chaque tick et ecrit actions.txt au format GDD
(une ligne "ACTION_F | ACTION_M" par tick, "END_GAME" en derniere ligne).

Usage :
    python PlayOnnx.py --map map.txt --elevation elevation.txt \\
        --onnx-f OnnxModels/ikotofosa.onnx --onnx-m OnnxModels/imahaki.onnx \\
        --actions actions.txt

Tests de non-regression (formule de score GDD, action_mask, format du
fichier de sortie) : voir test_play_onnx.py, executable via `pytest
test_play_onnx.py -q`. `--selftest` lance cette suite avant de jouer.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

from AlgoEnv import build_grid, build_scalars
from AlgoTrain import (
    DEFAULT_ENGINE_CONFIG,
    HEROES,
    N_GRID_CHANNELS,
    N_SCALARS,
    action_mask,
    read_elevation,
    read_map_header,
    resolve_file,
    validate_terrain,
)
from GameEngine import GameEngine

RESOURCE_LOW_TICKS_LIMIT = 10  # meme seuil que AlgoEnv._RESOURCE_LOW_TICKS_LIMIT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rejoue AlgoGames 2 avec les modeles ONNX F et M (pas de dependance a sb3-contrib)."
    )
    parser.add_argument("--map", required=True, dest="map_path")
    parser.add_argument("--elevation", required=True, dest="elevation_path")
    parser.add_argument("--actions", default="actions.txt", help="Fichier de sortie (format GDD).")
    parser.add_argument("--onnx-f", default="OnnxModels/ikotofosa.onnx")
    parser.add_argument("--onnx-m", default="OnnxModels/imahaki.onnx")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--max-ticks", type=int, default=None,
                         help="Surcharge le temps limite lu dans map.txt.")
    parser.add_argument("--tick-delay", type=float, default=0.0,
                         help="Pause en secondes entre chaque tick (GDD: 1 tick = 0.2s en temps reel).")
    parser.add_argument("--log-every", type=int, default=1,
                         help="Reaffiche la map tous les N ticks (defaut: chaque tick).")
    parser.add_argument("--no-render", action="store_true",
                         help="N'affiche que la ligne de stats a chaque tick, pas la map ascii.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--selftest", action="store_true",
                         help="Lance test_play_onnx.py (pytest) avant de jouer ; arrete si echec.")
    return parser.parse_args()


def load_session(path: str, device: str) -> ort.InferenceSession:
    onnx_path = resolve_file(path, "modele ONNX")
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]
    return ort.InferenceSession(str(onnx_path), providers=providers)


def pick_action(session: ort.InferenceSession, action_names: list[str],
                 grid: np.ndarray, scalars: np.ndarray, valid_mask: np.ndarray) -> str:
    """logits bruts (contrat Exports.py, pas d'argmax cote export) -> argmax
    masque, equivalent a MaskablePPO.predict(..., action_masks=...)."""
    logits = session.run(
        None,
        {"map": grid[None].astype(np.float32), "stats": scalars[None].astype(np.float32)},
    )[0][0]
    masked = np.where(valid_mask, logits, -np.inf)
    if not np.isfinite(masked).any():
        # Garde-fou : action_mask() garantit toujours >=1 action valide (WAIT
        # ou HACK_* selon on_engine), ce cas ne devrait jamais arriver.
        return "WAIT" if "WAIT" in action_names else action_names[0]
    return action_names[int(np.argmax(masked))]


def render_map(engine: GameEngine) -> str:
    """Rendu ASCII courant (terrain + coffres/pierres/machines/heros), lecture
    seule : ne mute pas l'etat du moteur."""
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


def format_actions_lines(pairs: list[tuple[str, str]]) -> list[str]:
    """Format GDD : "ACTION_F | ACTION_M" par tick, END_GAME en derniere ligne."""
    lines = [f"{a_f} | {a_m}" for a_f, a_m in pairs]
    lines.append("END_GAME")
    return lines


def run_selftest() -> bool:
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "test_play_onnx.py", "-q"],
        cwd=str(Path(__file__).resolve().parent),
    )
    return result.returncode == 0


def main() -> int:
    args = parse_args()

    if args.selftest:
        print("[selftest] pytest test_play_onnx.py ...")
        if not run_selftest():
            print("[selftest] echec -> arret avant de jouer.")
            return 1
        print("[selftest] OK")

    map_path = resolve_file(args.map_path, "map.txt")
    elevation_path = resolve_file(args.elevation_path, "elevation.txt")
    height, width, max_time, ascii_rows = read_map_header(map_path)
    elevation = read_elevation(elevation_path, height, width)
    validate_terrain(ascii_rows, elevation)
    max_ticks = args.max_ticks or max_time

    # Meme engine_config que celui utilise a l'entrainement (AlgoEnv.py via
    # AlgoTrain.make_single_env) : garantit un encodage 15 canaux identique.
    engine_config = dict(DEFAULT_ENGINE_CONFIG)
    engine = GameEngine(ascii_rows, elevation, max_time, engine_config=engine_config, seed=args.seed)

    sessions = {"F": load_session(args.onnx_f, args.device), "M": load_session(args.onnx_m, args.device)}
    action_names = {h: HEROES[h]["actions"] for h in ("F", "M")}

    pairs: list[tuple[str, str]] = []

    print(f"[play] map {width}x{height}, temps limite={max_time} (max-ticks={max_ticks})")
    if not args.no_render:
        print(render_map(engine))
    print(f"[tick 0] score={engine.official_score()}")

    tick = 0
    while tick < max_ticks:
        chosen: dict[str, str] = {}
        for h in ("F", "M"):
            grid_obs = build_grid(engine, h, elevation, engine_config, N_GRID_CHANNELS)
            scalars_obs = build_scalars(engine, h, N_SCALARS)
            mask = action_mask(engine.on_engine[h] is not None, action_names[h])
            chosen[h] = pick_action(sessions[h], action_names[h], grid_obs, scalars_obs, mask)

        engine.step(chosen)
        tick += 1
        pairs.append((chosen["F"], chosen["M"]))

        cleared = not engine.stones and not engine.chests
        resource_low = (
            engine.resource_low_ticks["F"] >= RESOURCE_LOW_TICKS_LIMIT
            or engine.resource_low_ticks["M"] >= RESOURCE_LOW_TICKS_LIMIT
        )
        should_log = tick % max(1, args.log_every) == 0 or cleared or resource_low or tick >= max_ticks

        if should_log:
            if not args.no_render:
                print(f"\n=== tick {tick} ===")
                print(render_map(engine))
            print(
                f"[tick {tick}] F={chosen['F']:<12} M={chosen['M']:<12} "
                f"stamina(F/M)={engine.stamina['F']:.1f}/{engine.stamina['M']:.1f} "
                f"batt(F/M)={engine.battery['F']:.1f}/{engine.battery['M']:.1f} "
                f"stones={engine.stones_collected} hidden={engine.chests_hidden} "
                f"score={engine.official_score()}"
            )

        if args.tick_delay > 0:
            time.sleep(args.tick_delay)

        if cleared or resource_low:
            print(f"[play] fin anticipee au tick {tick} ({'objectifs atteints' if cleared else 'ressources epuisees'}).")
            break

    actions_path = Path(args.actions)
    actions_path.parent.mkdir(parents=True, exist_ok=True)
    actions_path.write_text("\n".join(format_actions_lines(pairs)) + "\n", encoding="utf-8")

    print(f"\n[play] {tick} ticks joues, actions ecrites dans {actions_path}")
    print(f"[play] score final officiel (GDD) = {engine.official_score()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
