"""Recherche Optuna d'hyperparametres MaskablePPO.

Meme jeu de flags que AlgoTrain.py -> lancable en parallele pour F et M :

CUDA_VISIBLE_DEVICES=0 python Optuna.py --hero F \\
  --map map.txt --elevation elevation.txt \\
  --output ModelStates/ikotofosa --device cuda --augmentation mirror_horizontal \\
  --timesteps 10000 --n-envs 8 \\
  > optuna_log_F.txt 2>&1 &
CUDA_VISIBLE_DEVICES=1 python Optuna.py --hero M \\
  --map map.txt --elevation elevation.txt \\
  --output ModelStates/imahaki --device cuda --augmentation mirror_horizontal \\
  --timesteps 10000 --n-envs 8 \\
  > optuna_log_M.txt 2>&1 &
wait
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import optuna
from sb3_contrib import MaskablePPO                                      # MIGRATION MASKABLEPPO
from sb3_contrib.common.maskable.utils import get_action_masks           # MIGRATION MASKABLEPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from AlgoTrain import (
    AUGMENTATIONS,
    AlgoGamesEnv,
    HEROES,
    build_policy_kwargs,                                                # MIGRATION MASKABLEPPO
    find_factory,
    hyperparams_path,                                                   # dossier partage avec AlgoTrain.py
    import_module,
    make_single_env,
    read_elevation,
    read_map_header,
    resolve_file,
    validate_terrain,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recherche Optuna d'hyperparametres MaskablePPO (un hero a la fois)."
    )
    parser.add_argument(
        "--hero", required=True, choices=("F", "M"),
        help="Hero a optimiser : F=Ikotofosa (14 actions), M=Imahaki (13 actions).",
    )
    parser.add_argument("--map", required=True, dest="map_path")
    parser.add_argument("--elevation", required=True, dest="elevation_path")
    parser.add_argument(
        "--output", required=True,
        help="Non utilise par Optuna.py ; garde pour la parite de commande avec AlgoTrain.py "
             "(le fichier resultat va toujours dans OptunaParams/, cf. hyperparams_path()).",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--augmentation", choices=("random", "all", *AUGMENTATIONS), default="random")
    parser.add_argument("--timesteps", type=int, default=50_000, help="Budget d'entrainement par trial.")
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--n-eval-episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_factory():
    """Reprend le meme fallback que AlgoTrain.py : AlgoEnv.py si present."""
    algo_env_path = Path(__file__).resolve().parent / "AlgoEnv.py"
    if algo_env_path.is_file():
        module = import_module(algo_env_path, "algogames_env_optuna")
        return find_factory(module)
    return AlgoGamesEnv


def make_vec_env(factory, map_path, elevation_path, hero, augmentation, seed, n_envs):
    env_fns = [
        make_single_env(factory, map_path, elevation_path, hero, augmentation, seed, rank)
        for rank in range(n_envs)
    ]
    return VecMonitor(DummyVecEnv(env_fns))


def build_objective(args: argparse.Namespace, factory, map_path: Path, elevation_path: Path):
    n_actions = HEROES[args.hero]["n_actions"]

    def objective(trial: optuna.Trial) -> float:
        # 1. Echantillonnage des hyperparametres
        lr = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
        n_steps = trial.suggest_categorical("n_steps", [64, 128, 256])
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
        ent_coef = trial.suggest_float("ent_coef", 1e-4, 0.05, log=True)

        # rollout_size doit rester divisible par batch_size (meme regle qu'AlgoTrain.py)
        rollout_size = n_steps * args.n_envs
        trial_batch_size = min(batch_size, rollout_size)
        while rollout_size % trial_batch_size != 0 and trial_batch_size > 1:
            trial_batch_size -= 1

        # 2. Environnement vectorise + masquage d'actions (meme pipeline qu'AlgoTrain.py)
        train_env = make_vec_env(
            factory, map_path, elevation_path, args.hero,
            args.augmentation, args.seed + trial.number, args.n_envs,
        )

        # 3. Modele MaskablePPO
        model = MaskablePPO(                                             # MIGRATION MASKABLEPPO
            "MultiInputPolicy",                                          # MIGRATION MASKABLEPPO
            train_env,
            learning_rate=lr,
            n_steps=n_steps,
            batch_size=trial_batch_size,
            ent_coef=ent_coef,
            policy_kwargs=build_policy_kwargs(n_actions),                # meme extracteur qu'AlgoTrain.py
            device=args.device,
            seed=args.seed,
            verbose=0,
        )

        try:
            # 4. Entrainement d'evaluation (budget = --timesteps)
            model.learn(total_timesteps=args.timesteps)

            # 5. Calcul du score moyen sur un env d'eval dedie (augmentation identity)
            eval_env = make_vec_env(
                factory, map_path, elevation_path, args.hero,
                "identity", args.seed + 100_000, 1,
            )
            try:
                # cf. diagnostic Cause 4 : sum(rewards)/len(rewards) seul est
                # degenere (80% des essais tombent sur le meme score car la
                # politique WAIT-permanente est deterministe et son retour est
                # quasi constant). On distingue explicitement un essai qui
                # n'accomplit rien d'un essai qui progresse, meme si leurs
                # rewards moyens se ressemblent.
                rewards, chests, stones, wait_ratios, invalid_ratios = [], [], [], [], []
                for _ in range(args.n_eval_episodes):
                    obs = eval_env.reset()
                    done, total_r, last_info = False, 0.0, {}
                    while not done:
                        action_masks = get_action_masks(eval_env)        # MIGRATION MASKABLEPPO
                        action, _ = model.predict(
                            obs, deterministic=True, action_masks=action_masks,
                        )
                        obs, r, dones, infos = eval_env.step(action)
                        total_r += float(r[0])
                        done = bool(dones[0])
                        last_info = infos[0]
                    rewards.append(total_r)
                    chests.append(last_info.get("chests_hidden", 0))
                    stones.append(last_info.get("stones_collected", 0))
                    wait_ratios.append(last_info.get("wait_ratio", 0.0))
                    invalid_ratios.append(last_info.get("invalid_ratio", 0.0))
            finally:
                eval_env.close()
        finally:
            train_env.close()

        n = len(rewards)
        # Le critere Optuna privilegie le retour aligne sur le score officiel.
        # Les ratios servent uniquement de departage faible : ils ne doivent pas
        # dominer une pierre ou un coffre effectivement obtenu.
        score = (
            sum(rewards) / n
            + 0.01 * (sum(chests) / n)
            + 0.001 * (sum(stones) / n)
            - 0.01 * (sum(wait_ratios) / n)
            - 0.02 * (sum(invalid_ratios) / n)
        )
        trial.set_user_attr("mean_reward", sum(rewards) / n)
        trial.set_user_attr("mean_chests_hidden", sum(chests) / n)
        trial.set_user_attr("mean_stones_collected", sum(stones) / n)
        return score

    return objective


def main() -> None:
    args = parse_args()

    map_path = resolve_file(args.map_path, "map.txt")
    elevation_path = resolve_file(args.elevation_path, "elevation.txt")
    height, width, _max_time, ascii_rows = read_map_header(map_path)
    elevation = read_elevation(elevation_path, height, width)
    validate_terrain(ascii_rows, elevation)

    factory = load_factory()

    # Meme fichier que celui lu par AlgoTrain.py (OptunaParams/best_hyperparams_<hero>.json).
    best_params_path = hyperparams_path(args.hero)

    study = optuna.create_study(direction="maximize", study_name=f"maskableppo_{args.hero}")
    study.optimize(build_objective(args, factory, map_path, elevation_path), n_trials=args.n_trials)

    print(f"[optuna][{args.hero}] Meilleurs hyperparametres trouves :", study.best_params)

    with best_params_path.open("w", encoding="utf-8") as f:
        json.dump(study.best_params, f, indent=4)

    print(f"[optuna][{args.hero}] Parametres sauvegardes dans {best_params_path}")


if __name__ == "__main__":
    main()