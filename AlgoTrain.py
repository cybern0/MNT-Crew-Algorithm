#!/usr/bin/env python3
"""AlgoTrain.py — entrainement MaskablePPO, un modele par hero.

    python AlgoTrain.py --hero F --map map.txt --elevation elevation.txt \
        --output ModelStates/ikotofosa

Plusieurs cartes = variation reelle de topologie :

    python AlgoTrain.py --hero F --map m1.txt m2.txt --elevation e1.txt e2.txt ...

Reglages imposes par la recompense sparse (cf. AlgoEnv) :
  gamma=0.999   : un +6.0 terminal escompte a 0.74 sur 300 pas (0.995 -> 0.22,
                  invisible face au -0.02/tick) ;
  n_steps=2048  : a 1 % de succes, un rollout de 128 pas ne contient aucun
                  evenement positif -> gradient = bruit pur ;
  ent_coef      : 0.05 -> 0.005 decroissant, sinon on maintient du bruit
                  alors qu'une bonne politique emerge ;
  curriculum    : reverse curriculum sur les etats de depart, seul levier
                  d'exploration qui ne deforme pas la politique optimale.
"""
from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

from AlgoSpec import CURRICULUM_PHASES, HEROES, N_GRID_CHANNELS, N_SCALARS, load_maps
from AlgoEnv import AlgoEnv


class GridScalarExtractor(BaseFeaturesExtractor):
    """CNN 15 canaux + MLP scalaires. Le pooling s'arrete a 5x5 : un
    AdaptiveAvgPool2d((1,1)) ecraserait chaque canal en une moyenne et
    detruirait la position relative des objectifs."""

    def __init__(self, observation_space, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        self.cnn = nn.Sequential(
            nn.Conv2d(N_GRID_CHANNELS, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((5, 5)), nn.Flatten(),
        )
        self.mlp = nn.Sequential(
            nn.Linear(N_SCALARS, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU(),
        )
        self.combined = nn.Sequential(nn.Linear(64 * 25 + 32, features_dim), nn.ReLU())

    def forward(self, obs):
        return self.combined(torch.cat([self.cnn(obs["grid"]), self.mlp(obs["scalars"])], dim=1))


class MaskWrapper(gym.Wrapper):
    """Expose action_masks() et set_curriculum_phase() au niveau le plus
    externe : c'est ce que get_action_masks()/env_method() vont chercher.
    On passe par .unwrapped, sans dependre du forwarding d'attributs des
    wrappers gymnasium."""

    def action_masks(self) -> np.ndarray:
        return self.env.unwrapped.action_masks()

    def set_curriculum_phase(self, phase: int) -> None:
        self.env.unwrapped.set_curriculum_phase(phase)


def make_env_fn(maps, hero, seed, rank, phase):
    def _init():
        env = AlgoEnv(maps, hero=hero, curriculum_phase=phase, seed=seed + rank)
        env = Monitor(env)
        env = MaskWrapper(env)
        env.reset(seed=seed + rank)
        return env
    return _init


class EntropyDecay(BaseCallback):
    def __init__(self, start: float, end: float, total: int):
        super().__init__()
        self.start, self.end, self.total = start, end, max(1, total)

    def _on_step(self) -> bool:
        frac = min(1.0, self.num_timesteps / self.total)
        self.model.ent_coef = self.start + frac * (self.end - self.start)
        self.logger.record("train/ent_coef", self.model.ent_coef)
        return True


class ReverseCurriculum(BaseCallback):
    """Avance de phase quand le taux de succes est suffisant (ou apres un
    plafond de pas). Chaque phase fournit a la value function des etats dont
    elle connait deja la valeur ; la suivante ne demande qu'un pas de plus en
    arriere. C'est ce qui remplace un gradient BFS, sans mentir a l'agent."""

    def __init__(self, min_steps: int, max_steps: int, threshold: float, window: int = 50):
        super().__init__()
        self.min_steps, self.max_steps = min_steps, max_steps
        self.threshold = threshold
        self.successes: deque[float] = deque(maxlen=window)
        self.phase, self.phase_start = 0, 0

    def _on_training_start(self) -> None:
        self.model.get_env().env_method("set_curriculum_phase", self.phase)

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                won = info.get("hero_stones", 0) or info.get("hero_chests", 0)
                self.successes.append(1.0 if won else 0.0)
        rate = float(np.mean(self.successes)) if self.successes else 0.0
        elapsed = self.num_timesteps - self.phase_start
        ready = (elapsed >= self.min_steps
                 and len(self.successes) == self.successes.maxlen
                 and rate >= self.threshold)
        if self.phase < len(CURRICULUM_PHASES) - 1 and (ready or elapsed >= self.max_steps):
            self.phase += 1
            self.phase_start = self.num_timesteps
            self.successes.clear()
            self.model.get_env().env_method("set_curriculum_phase", self.phase)
            print(f"[curriculum] phase -> {self.phase} "
                  f"({CURRICULUM_PHASES[self.phase]}) a {self.num_timesteps} pas")
        self.logger.record("curriculum/phase", self.phase)
        self.logger.record("curriculum/success_rate", rate)
        return True


class ScoreLogger(BaseCallback):
    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" not in info:
                continue
            self.logger.record_mean("game/official_score", float(info["official_score"]))
            self.logger.record_mean("game/hero_stones", float(info["hero_stones"]))
            self.logger.record_mean("game/hero_chests", float(info["hero_chests"]))
            self.logger.record_mean("game/chests_destroyed",
                                    float(info["hero_chests_destroyed"]))
            self.logger.record_mean("game/ticks", float(info["tick"]))
        return True


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Entraine MaskablePPO sur AlgoGames 2.")
    p.add_argument("--hero", required=True, choices=("F", "M"))
    p.add_argument("--map", required=True, nargs="+", dest="map_paths")
    p.add_argument("--elevation", required=True, nargs="+", dest="elevation_paths")
    p.add_argument("--output", required=True, help="Chemin sans extension.")
    p.add_argument("--timesteps", type=int, default=2_000_000)
    p.add_argument("--n-envs", type=int, default=16)
    p.add_argument("--vec", choices=("dummy", "subproc"), default="dummy")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--n-steps", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--gamma", type=float, default=0.999)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--ent-coef", type=float, default=0.05)
    p.add_argument("--ent-coef-final", type=float, default=0.005)
    p.add_argument("--no-curriculum", action="store_true")
    p.add_argument("--phase-min-steps", type=int, default=100_000)
    p.add_argument("--phase-max-steps", type=int, default=400_000)
    p.add_argument("--phase-threshold", type=float, default=0.5)
    p.add_argument("--eval-freq", type=int, default=0, help="0 = desactive.")
    p.add_argument("--checkpoint-freq", type=int, default=200_000)
    p.add_argument("--resume")
    p.add_argument("--check-env", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    hero = args.hero
    maps = load_maps(args.map_paths, args.elevation_paths)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    output = Path(args.output).expanduser().resolve().with_suffix(".zip")
    output.parent.mkdir(parents=True, exist_ok=True)

    print(f"[train] hero={HEROES[hero]['name']} ({hero}) "
          f"n_actions={HEROES[hero]['n_actions']} cartes={[m.name for m in maps]}")

    if args.check_env:
        probe = make_env_fn(maps, hero, args.seed, 0, len(CURRICULUM_PHASES) - 1)()
        try:
            check_env(probe, warn=True)
        finally:
            probe.close()

    start_phase = len(CURRICULUM_PHASES) - 1 if args.no_curriculum else 0
    vec_cls = SubprocVecEnv if args.vec == "subproc" else DummyVecEnv
    train_env = VecMonitor(vec_cls([
        make_env_fn(maps, hero, args.seed, rank, start_phase)
        for rank in range(args.n_envs)
    ]))

    rollout = args.n_steps * args.n_envs
    batch = min(args.batch_size, rollout)
    while rollout % batch and batch > 1:
        batch -= 1

    if args.resume:
        model = MaskablePPO.load(args.resume, env=train_env, device=args.device)
    else:
        model = MaskablePPO(
            "MultiInputPolicy", train_env,
            learning_rate=args.learning_rate, n_steps=args.n_steps, batch_size=batch,
            gamma=args.gamma, gae_lambda=args.gae_lambda, ent_coef=args.ent_coef,
            policy_kwargs={
                "features_extractor_class": GridScalarExtractor,
                "features_extractor_kwargs": {"features_dim": 128},
                "net_arch": {"pi": [64], "vf": [64]},
            },
            device=args.device, seed=args.seed, verbose=1,
            tensorboard_log=str(output.parent / f"tensorboard_{hero}"),
        )

    callbacks: list[BaseCallback] = [
        ScoreLogger(),
        EntropyDecay(args.ent_coef, args.ent_coef_final, args.timesteps),
    ]
    if not args.no_curriculum:
        callbacks.append(ReverseCurriculum(
            args.phase_min_steps, args.phase_max_steps, args.phase_threshold))
    if args.checkpoint_freq > 0:
        callbacks.append(CheckpointCallback(
            save_freq=max(args.checkpoint_freq // args.n_envs, 1),
            save_path=str(output.parent / f"checkpoints_{hero}"),
            name_prefix=output.stem))

    eval_env = None
    if args.eval_freq > 0:
        # Evaluation toujours en phase finale (positions du GDD).
        eval_env = VecMonitor(DummyVecEnv([
            make_env_fn(maps, hero, args.seed + 10_000, 0, len(CURRICULUM_PHASES) - 1)]))
        callbacks.append(MaskableEvalCallback(
            eval_env, best_model_save_path=str(output.parent / f"best_{hero}"),
            log_path=str(output.parent / f"best_{hero}"),
            eval_freq=max(args.eval_freq // args.n_envs, 1),
            n_eval_episodes=5, deterministic=True, use_masking=True))

    try:
        model.learn(total_timesteps=args.timesteps,
                    callback=CallbackList(callbacks),
                    reset_num_timesteps=not bool(args.resume))
        model.save(str(output.with_suffix("")))
        print(f"[train] modele sauvegarde : {output}")
    finally:
        train_env.close()
        if eval_env is not None:
            eval_env.close()


if __name__ == "__main__":
    main()
