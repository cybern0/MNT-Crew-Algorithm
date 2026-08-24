"""Exports.py — exporte les archives MaskablePPO en ONNX.

    python Exports.py --hero F      # ModelStates/ikotofosa.zip -> OnnxModels/ikotofosa.onnx
    python Exports.py --hero both

Contrat de sortie (mirroir du runtime C#) :
    entrees  : map [batch, 15, 30, 30], stats [batch, 6]  (map AVANT stats)
    sortie   : action = LOGITS bruts [batch, n_actions]
L'argmax est applique cote appelant, qui verifie que la taille de sortie vaut
14 (F) ou 13 (M) : renvoyer un argmax donnerait [batch] et casserait ce check.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from sb3_contrib import MaskablePPO

from AlgoSpec import HEROES, MAX_HEIGHT, MAX_WIDTH, N_GRID_CHANNELS, N_SCALARS

MODEL_DIR = Path("ModelStates")
ONNX_DIR = Path("OnnxModels")


class ONNXPolicy(nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.features_extractor = policy.features_extractor
        self.mlp_extractor = policy.mlp_extractor
        self.action_net = policy.action_net

    def forward(self, map_, stats):
        features = self.features_extractor({"grid": map_, "scalars": stats})
        return self.action_net(self.mlp_extractor.forward_actor(features))


def _n_outputs(action_net) -> int:
    return (action_net[-1].out_features if isinstance(action_net, nn.Sequential)
            else action_net.out_features)


def export(hero: str, model_dir: Path, onnx_dir: Path) -> None:
    name = HEROES[hero]["name"]
    expected = HEROES[hero]["n_actions"]
    src = model_dir / f"{name}.zip"
    if not src.is_file():
        src = model_dir / name
    if not src.exists():
        raise FileNotFoundError(f"archive introuvable pour {name} dans {model_dir}")

    # CPU force : torch.onnx.export trace en eager, un poids reste sur CUDA
    # leve "Expected all tensors to be on the same device".
    model = MaskablePPO.load(str(src), device="cpu")
    policy = model.policy
    policy.eval()
    policy.to("cpu")

    actual = _n_outputs(policy.action_net)
    if actual != expected:
        raise ValueError(f"{name} : modele a {actual} actions, contrat en attend {expected}.")

    wrapper = ONNXPolicy(policy).to("cpu").eval()
    onnx_dir.mkdir(parents=True, exist_ok=True)
    out = onnx_dir / f"{name}.onnx"
    torch.onnx.export(
        wrapper,
        (torch.zeros(1, N_GRID_CHANNELS, MAX_HEIGHT, MAX_WIDTH),
         torch.zeros(1, N_SCALARS)),
        str(out),
        input_names=["map", "stats"], output_names=["action"],
        dynamic_axes={"map": {0: "batch"}, "stats": {0: "batch"}, "action": {0: "batch"}},
        opset_version=17,
    )
    print(f"[export] {src} -> {out} (n_actions={actual})")


def main() -> int:
    p = argparse.ArgumentParser(description="Export ONNX AlgoGames 2.")
    p.add_argument("--hero", required=True, choices=("F", "M", "both"))
    p.add_argument("--dir", default=str(MODEL_DIR))
    p.add_argument("--outdir", default=str(ONNX_DIR))
    args = p.parse_args()

    heroes = ("F", "M") if args.hero == "both" else (args.hero,)
    failures = 0
    for hero in heroes:
        try:
            export(hero, Path(args.dir), Path(args.outdir))
        except Exception as exc:
            print(f"[error] {hero} : {exc}", file=sys.stderr)
            failures += 1
    return 2 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
