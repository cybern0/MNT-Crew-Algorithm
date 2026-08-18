"""Exports.py — Exporte Ikotofosa (14 actions) et Imahaki (13 actions) en ONNX.

Lit ModelState/ikotofosa.zip et ModelState/imahaki.zip, ecrit les .onnx
au meme endroit.

Le catalogue d'actions est aligne avec AlgoTrain.py :

    Ikotofosa (F) : 14 actions (HACK_FILL incluse, excavation de trous)
    Imahaki  (M) : 13 actions (HACK_CUT automatique chez les grapplers,
                              donc non emis)

Important : le modele et le trainer utilisent une carte 13 canaux
(11 canaux one-hot de tuile + elevation absolue + elevation relative).
Cet exporteur emet donc `map` avec shape [1, 13, 30, 30] pour matcher
le modele entraine. Le preprocesseur C# doit lui aussi produire 13 canaux.

Usage :

    python Exports.py --hero F                 # exporte ikotofosa.onnx (14)
    python Exports.py --hero M                 # exporte imahaki.onnx  (13)
    python Exports.py --hero both              # exporte les deux
    python Exports.py --export --all          # exporte tous les .zip
    python Exports.py --list                  # liste les archives disponibles
"""
import os
import sys
import torch
import torch.nn as nn
from stable_baselines3 import PPO

MODEL_STATE_DIR = "ModelStates"
ONNX_OUT_DIR = "OnnxModels"

# ---------------------------------------------------------------------------
# Catalogue des modeles attendus : nom d'archive -> nombre d'actions.
# Doit matcher HEROES dans AlgoTrain.py.
# ---------------------------------------------------------------------------
MODEL_SPECS: dict[str, dict] = {
    "ikotofosa": {
        "hero": "F",
        "n_actions": 14,
        "description": "Ikotofosa - 14 actions (HACK_FILL pour excavateurs)",
        "action_names": [
            "UP", "DOWN", "LEFT", "RIGHT",
            "WAIT",
            "PUSH_UP", "PUSH_DOWN", "PUSH_LEFT", "PUSH_RIGHT",
            "HACK",
            "HACK_MOVE", "HACK_FILL", "HACK_CW", "HACK_CCW",
        ],
    },
    "imahaki": {
        "hero": "M",
        "n_actions": 13,
        "description": "Imahaki - 13 actions (HACK_CUT auto sur grapplers)",
        "action_names": [
            "UP", "DOWN", "LEFT", "RIGHT",
            "WAIT",
            "PUSH_UP", "PUSH_DOWN", "PUSH_LEFT", "PUSH_RIGHT",
            "HACK",
            "HACK_MOVE", "HACK_CW", "HACK_CCW",
        ],
    },
}


class ONNXPolicyWrapper(nn.Module):
    """map avant stats : doit matcher l'ordre lu cote C#
    (_session.InputMetadata.Keys[0]=map, [1]=stats).

    Le wrapper ne depend pas du nombre d'actions : il appelle action_net
    qui est dimensionnee automatiquement par SB3 a partir de
    policy.action_space.n. On renvoie le argmax (action discrete) ; si
    vous avez besoin des probas brutes pour un sampling temperature, il
    faut remplacer torch.argmax par logits.
    """

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


def _resolve_model_path(name: str, model_state_dir: str) -> str:
    """Trouve le .zip du modele (ou le fichier sans extension si SB3 l'a ecrit)."""
    base_path = os.path.join(model_state_dir, name)
    zip_path = base_path + ".zip"

    if os.path.exists(base_path) and not os.path.isdir(base_path):
        return base_path
    if os.path.exists(zip_path):
        return zip_path

    found = os.listdir(model_state_dir) if os.path.exists(model_state_dir) else []
    raise FileNotFoundError(
        f"Impossible de trouver l'archive du modele '{name}' dans {model_state_dir}.\n"
        f"Tente : '{base_path}' et '{zip_path}'.\n"
        f"Fichiers trouves : {found}"
    )


def _verify_action_count(policy, expected_n_actions: int, name: str) -> None:
    """Verifie que action_net produit exactement `expected_n_actions` sorties.

    Cela detecte une incoherence entre le modele charge (entraine avec
    Discrete(N)) et le contrat ONNX attendu cote C#. Sans cette verification,
    on pourrait exporter un 14-actions alors que le C# attend 13, et l'erreur
    n'apparaitrait qu'a l'inference.
    """
    actual = policy.action_net[-1].out_features if isinstance(
        policy.action_net, nn.Sequential
    ) else policy.action_net.out_features

    if actual != expected_n_actions:
        raise ValueError(
            f"[export] {name} : le modele contient {actual} actions, "
            f"mais MODEL_SPECS en attend {expected_n_actions}. "
            f"Reentrainez le modele avec --hero {MODEL_SPECS[name]['hero']} "
            f"ou mettez a jour MODEL_SPECS."
        )


def export_model(name: str, n_actions: int | None = None) -> None:
    """Exporte un modele SB3 en ONNX.

    Args:
        name: nom de base de l'archive (sans .zip). Ex : "ikotofosa".
        n_actions: nombre d'actions attendues. Si None, on lit MODEL_SPECS.
            Si MODEL_SPECS ne contient pas `name`, on essaie de le deduire
            de policy.action_net.out_features (sans verification).
    """
    if n_actions is None and name in MODEL_SPECS:
        n_actions = MODEL_SPECS[name]["n_actions"]

    load_path = _resolve_model_path(name, MODEL_STATE_DIR)
    onnx_path = os.path.join(ONNX_OUT_DIR, f"{name}.onnx")
    os.makedirs(ONNX_OUT_DIR, exist_ok=True)

    model = PPO.load(load_path)
    policy = model.policy
    policy.eval()

    if n_actions is not None:
        _verify_action_count(policy, n_actions, name)

    wrapper = ONNXPolicyWrapper(policy)

    # 13 canaux : 11 canaux one-hot de tuile + elevation absolue + relative.
    map_ = torch.zeros(1, 13, 30, 30)
    # 6 scalaires : stamina, batterie, temps, X, Y, isOnEngine.
    stats = torch.zeros(1, 6)

    torch.onnx.export(
        wrapper, (map_, stats), onnx_path,
        input_names=["map", "stats"], output_names=["action"],
        dynamic_axes={"map": {0: "batch"}, "stats": {0: "batch"}},
        opset_version=17, dynamo=False,
    )

    actual_n = (
        policy.action_net[-1].out_features
        if isinstance(policy.action_net, nn.Sequential)
        else policy.action_net.out_features
    )
    print(
        f"[export] {load_path} -> {onnx_path} "
        f"(n_actions={actual_n}, attendu={n_actions})"
    )


def _list_archives(model_state_dir: str) -> list[str]:
    if not os.path.exists(model_state_dir):
        return []
    return sorted(
        os.path.splitext(entry)[0]
        for entry in os.listdir(model_state_dir)
        if entry.lower().endswith(".zip")
    )


def _normalize_name(raw: str) -> str:
    """Accepte 'ikotofosa', 'IKOTOFOSA', 'F', 'Imahaki', 'M' et renvoie
    toujours la cle canonique ('ikotofosa' ou 'imahaki')."""
    lowered = raw.lower().strip()
    aliases = {
        "f": "ikotofosa",
        "ikotofosa": "ikotofosa",
        "m": "imahaki",
        "imahaki": "imahaki",
    }
    if lowered not in aliases:
        return lowered  # on laisse tomber ; le caller levera une erreur claire
    return aliases[lowered]


def main() -> int:
    import argparse

    global MODEL_STATE_DIR, ONNX_OUT_DIR  # noqa: PLW0603

    parser = argparse.ArgumentParser(
        description="Exporte les archives SB3 en ONNX pour AlgoGames 2."
    )
    parser.add_argument(
        "--hero",
        choices=("F", "M", "both"),
        help="Hero a exporter : F=Ikotofosa (14 actions), M=Imahaki (13 actions), "
             "both=les deux. Mutuellement exclusif avec --model/--all.",
    )
    parser.add_argument(
        "--model", "-m", action="append",
        help="Nom d'archive a exporter (sans .zip). Peut etre repete.",
    )
    parser.add_argument("--all", action="store_true",
                        help="Exporter toutes les .zip trouvees dans ModelStates/.")
    parser.add_argument("--list", action="store_true",
                        help="Lister les archives disponibles dans ModelStates/.")
    parser.add_argument("--dir", default=MODEL_STATE_DIR,
                        help="Repertoire contenant les archives (.zip).")
    parser.add_argument("--outdir", default=ONNX_OUT_DIR,
                        help="Repertoire de sortie pour les .onnx.")
    parser.add_argument("--export", action="store_true",
                        help="Effectuer reellement l'export. Sans ce flag, "
                             "seulement lister les candidats.")
    args = parser.parse_args()

    MODEL_STATE_DIR = args.dir
    ONNX_OUT_DIR = args.outdir

    candidates = _list_archives(MODEL_STATE_DIR)

    # Mode liste.
    if args.list or (not args.export and not args.model and not args.all and not args.hero):
        print("Archives trouvees dans", MODEL_STATE_DIR)
        for name in candidates:
            spec = MODEL_SPECS.get(name)
            if spec:
                print(f" - {name}  [{spec['n_actions']} actions]  {spec['description']}")
            else:
                print(f" - {name}  (hors MODEL_SPECS)")
        if not args.export:
            print("Lancez avec --export --hero F (ou M / both) pour exporter.")
        return 0

    # Resoudre la liste a exporter.
    to_export: list[str] = []

    if args.hero:
        if args.model or args.all:
            parser.error("--hero est mutuellement exclusif avec --model/--all.")
        if args.hero == "both":
            to_export = ["ikotofosa", "imahaki"]
        else:
            to_export = [_normalize_name(args.hero)]
    elif args.all:
        to_export = sorted(candidates)
    elif args.model:
        to_export = [_normalize_name(m) for m in args.model]
    else:
        parser.error("Specifiez --hero F|M|both, --model NAME ou --all.")

    if not to_export:
        print("Aucun modele a exporter.")
        return 1

    failures = 0
    for name in to_export:
        try:
            export_model(name)
        except FileNotFoundError as exc:
            print(f"[error] {name} : {exc}")
            failures += 1
        except ValueError as exc:
            # Incoherence du nombre d'actions : erreur bloquante a remonter.
            print(f"[error] {name} : {exc}", file=sys.stderr)
            failures += 1
        except Exception as exc:  # noqa: BLE001
            print(f"[error] echec export {name} : {exc}")
            failures += 1

    if failures:
        print(f"[done] {failures} export(s) en echec sur {len(to_export)}.")
        return 2
    print(f"[done] {len(to_export)} export(s) OK dans {ONNX_OUT_DIR}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
