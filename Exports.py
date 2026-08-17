"""Exports.py — Exporte Ikotofosa (14 actions) et Imahaki (13 actions) en ONNX.
Lit ModelState/ikotofosa.zip et ModelState/imahaki.zip, écrit les .onnx au même endroit.
"""
import os
import torch
import torch.nn as nn
from stable_baselines3 import PPO

MODEL_STATE_DIR = "ModelStates"
ONNX_OUT_DIR = "OnnxModels"


class ONNXPolicyWrapper(nn.Module):
    """map avant stats : doit matcher l'ordre lu côté C# (_session.InputMetadata.Keys[0]=map, [1]=stats)."""
    def __init__(self, policy):
        super().__init__()
        self.features_extractor = policy.features_extractor
        self.mlp_extractor = policy.mlp_extractor
        self.action_net = policy.action_net

    def forward(self, map_, stats):
        features = self.features_extractor({"grid": map_, "scalars": stats})
        latent_pi = self.mlp_extractor.forward_actor(features)
        logits = self.action_net(latent_pi)
        return torch.argmax(logits, dim=1)


def export_model(name, n_actions):
    zip_path = os.path.join(MODEL_STATE_DIR, f"{name}.zip")
    onnx_path = os.path.join(ONNX_OUT_DIR, f"{name}.onnx")
    os.makedirs(ONNX_OUT_DIR, exist_ok=True)

    model = PPO.load(zip_path)
    policy = model.policy
    policy.eval()
    wrapper = ONNXPolicyWrapper(policy)

    map_ = torch.zeros(1, 11, 30, 30)
    stats = torch.zeros(1, 6)

    torch.onnx.export(
        wrapper, (map_, stats), onnx_path,
        input_names=["map", "stats"], output_names=["action"],
        dynamic_axes={"map": {0: "batch"}, "stats": {0: "batch"}},
        opset_version=17, dynamo=False,
    )
    print(f"[export] {zip_path} -> {onnx_path} ({n_actions} actions attendues)")


if __name__ == "__main__":
    export_model("ikotofosa", 14)
    export_model("imahaki", 13)