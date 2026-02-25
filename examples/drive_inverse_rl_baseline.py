#!/usr/bin/env python3
"""
Inverse-RL baseline for single-agent PufferDrive using the `imitation` library.

Pipeline:
1) train_expert  -> Train an SB3 PPO expert in control_sdc_only mode.
2) collect_demos -> Roll out the expert and save trajectory demonstrations.
3) train_airl    -> Train AIRL with those demonstrations.

This baseline targets the setting where one agent is policy-controlled and
other scene actors follow replay/static behavior from logs.
"""

from __future__ import annotations

import argparse
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

from pufferlib.ocean.drive.drive import Drive


def _require_optional_deps() -> tuple[Any, Any, Any, Any]:
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.vec_env import DummyVecEnv
    except ImportError as exc:
        raise ImportError(
            "Missing stable-baselines3 dependency. Install with:\n"
            "  pip install stable-baselines3"
        ) from exc

    try:
        from imitation.algorithms.adversarial.airl import AIRL
        from imitation.data.types import Trajectory
        from imitation.rewards.reward_nets import BasicShapedRewardNet
    except ImportError as exc:
        raise ImportError(
            "Missing imitation dependency. Install with:\n"
            "  pip install imitation"
        ) from exc

    return PPO, Monitor, DummyVecEnv, (AIRL, Trajectory, BasicShapedRewardNet)


@dataclass
class DriveEnvConfig:
    map_dir: str
    num_maps: int
    episode_length: int
    goal_behavior: int
    action_type: str
    dynamics_model: str
    termination_mode: int = 1
    init_mode: str = "create_all_valid"
    control_mode: str = "control_sdc_only"
    num_agents: int = 1
    render_mode: str | None = None


class SingleAgentDriveEnv(gym.Env):
    """
    Gymnasium wrapper for PufferDrive in one-controlled-agent mode.

    Assumes Drive is configured with control_mode='control_sdc_only' and
    num_agents=1 so that the first row of obs/reward/done corresponds to the
    policy-controlled SDC agent.
    """

    metadata = {"render_modes": ["human", "rgb_array", None]}

    def __init__(self, cfg: DriveEnvConfig):
        super().__init__()
        self.cfg = cfg
        self._env = Drive(
            map_dir=cfg.map_dir,
            num_maps=cfg.num_maps,
            episode_length=cfg.episode_length,
            goal_behavior=cfg.goal_behavior,
            action_type=cfg.action_type,
            dynamics_model=cfg.dynamics_model,
            termination_mode=cfg.termination_mode,
            init_mode=cfg.init_mode,
            control_mode=cfg.control_mode,
            num_agents=cfg.num_agents,
            render_mode=cfg.render_mode,
        )

        self.observation_space = self._env.single_observation_space
        self._is_discrete = isinstance(self._env.single_action_space, gym.spaces.MultiDiscrete)
        if self._is_discrete:
            nvec = self._env.single_action_space.nvec
            if len(nvec) != 1:
                raise ValueError(
                    "Expected 1D MultiDiscrete for single-agent discrete control, "
                    f"got nvec={nvec}"
                )
            self.action_space = gym.spaces.Discrete(int(nvec[0]))
        else:
            self.action_space = self._env.single_action_space

    def _pack_action(self, action: np.ndarray | int | float) -> np.ndarray:
        if self._is_discrete:
            return np.asarray([[int(action)]], dtype=np.int32)

        arr = np.asarray(action, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[None, :]
        return arr

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is None:
            obs, _ = self._env.reset()
        else:
            obs, _ = self._env.reset(seed=seed)
        return obs[0].astype(np.float32), {}

    def step(self, action):
        packed_action = self._pack_action(action)
        obs, rewards, terminals, truncations, infos = self._env.step(packed_action)
        info = infos[0] if infos and isinstance(infos[0], dict) else {}
        return (
            obs[0].astype(np.float32),
            float(rewards[0]),
            bool(terminals[0]),
            bool(truncations[0]),
            info,
        )

    def close(self):
        self._env.close()


def _make_env_fn(cfg: DriveEnvConfig):
    def thunk():
        return SingleAgentDriveEnv(cfg)

    return thunk


def train_expert(args):
    PPO, Monitor, DummyVecEnv, _ = _require_optional_deps()
    cfg = DriveEnvConfig(
        map_dir=args.map_dir,
        num_maps=args.num_maps,
        episode_length=args.episode_length,
        goal_behavior=args.goal_behavior,
        action_type=args.action_type,
        dynamics_model=args.dynamics_model,
        termination_mode=args.termination_mode,
    )

    env = DummyVecEnv([lambda: Monitor(_make_env_fn(cfg)())])
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        verbose=1,
        seed=args.seed,
        device=args.device,
    )
    model.learn(total_timesteps=args.total_timesteps)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    model_path = Path(args.output_dir) / "expert_ppo.zip"
    model.save(str(model_path))
    print(f"Saved expert policy: {model_path}")


def collect_demos(args):
    PPO, _, _, (_, Trajectory, _) = _require_optional_deps()
    cfg = DriveEnvConfig(
        map_dir=args.map_dir,
        num_maps=args.num_maps,
        episode_length=args.episode_length,
        goal_behavior=args.goal_behavior,
        action_type=args.action_type,
        dynamics_model=args.dynamics_model,
        termination_mode=args.termination_mode,
    )

    env = SingleAgentDriveEnv(cfg)
    model = PPO.load(args.expert_path, device=args.device)

    trajectories = []
    for ep in range(args.num_episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        obs_seq = [obs]
        act_seq = []
        info_seq = []
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            next_obs, _, terminated, truncated, info = env.step(action)
            act_seq.append(np.array(action))
            info_seq.append(info)
            obs_seq.append(next_obs)
            obs = next_obs
            done = terminated or truncated

        traj = Trajectory(
            obs=np.asarray(obs_seq, dtype=np.float32),
            acts=np.asarray(act_seq),
            infos=np.asarray(info_seq, dtype=object),
            terminal=True,
        )
        trajectories.append(traj)
        print(f"Collected trajectory {ep + 1}/{args.num_episodes} (len={len(act_seq)})")

    env.close()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    demos_path = Path(args.output_dir) / "expert_trajectories.pkl"
    with open(demos_path, "wb") as f:
        pickle.dump(trajectories, f)
    print(f"Saved demonstrations: {demos_path}")


def train_airl(args):
    PPO, Monitor, DummyVecEnv, (AIRL, _, BasicShapedRewardNet) = _require_optional_deps()
    cfg = DriveEnvConfig(
        map_dir=args.map_dir,
        num_maps=args.num_maps,
        episode_length=args.episode_length,
        goal_behavior=args.goal_behavior,
        action_type=args.action_type,
        dynamics_model=args.dynamics_model,
        termination_mode=args.termination_mode,
    )

    with open(args.demos_path, "rb") as f:
        demonstrations = pickle.load(f)

    venv = DummyVecEnv([lambda: Monitor(_make_env_fn(cfg)())])
    gen_algo = PPO(
        "MlpPolicy",
        venv,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        verbose=1,
        seed=args.seed,
        device=args.device,
    )

    reward_net = BasicShapedRewardNet(venv.observation_space, venv.action_space)
    airl_trainer = AIRL(
        demonstrations=demonstrations,
        demo_batch_size=args.demo_batch_size,
        venv=venv,
        gen_algo=gen_algo,
        reward_net=reward_net,
        allow_variable_horizon=args.allow_variable_horizon,
    )
    airl_trainer.train(total_timesteps=args.total_timesteps)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    policy_path = Path(args.output_dir) / "airl_generator_policy.zip"
    gen_algo.save(str(policy_path))
    print(f"Saved AIRL generator policy: {policy_path}")


def build_parser():
    parser = argparse.ArgumentParser(description="Single-agent PufferDrive inverse-RL baseline (AIRL).")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common_flags(p):
        p.add_argument("--map-dir", type=str, default="resources/drive/binaries/training")
        p.add_argument("--num-maps", type=int, default=1024)
        p.add_argument("--episode-length", type=int, default=91)
        p.add_argument("--goal-behavior", type=int, default=0, choices=[0, 1, 2])
        p.add_argument(
            "--termination-mode",
            type=int,
            default=1,
            choices=[0, 1],
            help="0: fixed horizon at episode_length, 1: terminate after all agents reset",
        )
        p.add_argument("--action-type", type=str, default="discrete", choices=["discrete", "continuous"])
        p.add_argument("--dynamics-model", type=str, default="classic", choices=["classic", "jerk"])
        p.add_argument("--seed", type=int, default=42)
        p.add_argument("--device", type=str, default="auto")
        p.add_argument("--output-dir", type=str, default="experiments/drive_inverse_rl")

    def add_ppo_flags(p):
        p.add_argument("--learning-rate", type=float, default=3e-4)
        p.add_argument("--n-steps", type=int, default=1024)
        p.add_argument("--batch-size", type=int, default=256)
        p.add_argument("--n-epochs", type=int, default=10)
        p.add_argument("--gamma", type=float, default=0.99)
        p.add_argument("--gae-lambda", type=float, default=0.95)

    p_train_expert = subparsers.add_parser("train_expert", help="Train SB3 PPO expert policy.")
    add_common_flags(p_train_expert)
    add_ppo_flags(p_train_expert)
    p_train_expert.add_argument("--total-timesteps", type=int, default=1_000_000)

    p_collect = subparsers.add_parser("collect_demos", help="Collect demonstrations from trained expert.")
    add_common_flags(p_collect)
    p_collect.add_argument("--expert-path", type=str, required=True)
    p_collect.add_argument("--num-episodes", type=int, default=128)

    p_airl = subparsers.add_parser("train_airl", help="Train AIRL baseline from demonstrations.")
    add_common_flags(p_airl)
    add_ppo_flags(p_airl)
    p_airl.add_argument("--demos-path", type=str, required=True)
    p_airl.add_argument("--demo-batch-size", type=int, default=1024)
    p_airl.add_argument("--total-timesteps", type=int, default=1_000_000)
    p_airl.add_argument(
        "--allow-variable-horizon",
        default=True,
        action="store_true",
        help="Pass through to imitation AIRL for variable-length episodes.",
    )

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "train_expert":
        train_expert(args)
    elif args.command == "collect_demos":
        collect_demos(args)
    elif args.command == "train_airl":
        train_airl(args)
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
