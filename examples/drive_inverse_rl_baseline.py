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
import os
import pickle
import subprocess
import time
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


def _maybe_init_wandb(args, job_type: str):
    if not args.wandb:
        return None, None

    try:
        import wandb
    except ImportError as exc:
        raise ImportError(
            "Missing wandb dependency. Install with:\n"
            "  pip install wandb"
        ) from exc

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        group=args.wandb_group,
        name=args.wandb_run_name,
        job_type=job_type,
        tags=args.wandb_tags,
        config=vars(args),
        sync_tensorboard=True,
    )
    tensorboard_log = str(Path(args.output_dir) / "tensorboard" / job_type)
    return run, tensorboard_log


def _finish_wandb(run, **summary):
    if run is None:
        return

    for key, value in summary.items():
        run.summary[key] = value

    import wandb

    wandb.finish()


def _normalize_display(display_env: str) -> str:
    # x11grab expects display in host.display form (e.g., :99.0)
    return display_env if "." in display_env else f"{display_env}.0"


def _start_ffmpeg_capture(args):
    if args.save_video_path is None:
        return None

    display = os.environ.get("DISPLAY")
    if not display:
        raise RuntimeError(
            "DISPLAY is not set. For headless capture, run with xvfb-run, e.g.\n"
            'xvfb-run -s "-screen 0 1280x720x24" python ... render_policy --save-video-path <out.mp4>'
        )

    out_path = Path(args.save_video_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    input_display = _normalize_display(display)
    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "x11grab",
        "-video_size",
        args.video_size,
        "-framerate",
        str(args.video_fps),
        "-i",
        input_display,
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(out_path),
    ]
    proc = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return proc


def _stop_ffmpeg_capture(proc):
    if proc is None:
        return

    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


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

    def render(self):
        return self._env.render()


def _make_env_fn(cfg: DriveEnvConfig):
    def thunk():
        return SingleAgentDriveEnv(cfg)

    return thunk


def train_expert(args):
    PPO, Monitor, DummyVecEnv, _ = _require_optional_deps()
    run, tb_log_dir = _maybe_init_wandb(args, "train_expert")
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
    try:
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
            tensorboard_log=tb_log_dir,
        )

        model.learn(total_timesteps=args.total_timesteps)
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        model_path = Path(args.output_dir) / "expert_ppo.zip"
        model.save(str(model_path))
        print(f"Saved expert policy: {model_path}")
    finally:
        _finish_wandb(run, expert_model_path=str(Path(args.output_dir) / "expert_ppo.zip"))


def collect_demos(args):
    PPO, _, _, (_, Trajectory, _) = _require_optional_deps()
    run, _ = _maybe_init_wandb(args, "collect_demos")
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

    try:
        trajectories = []
        traj_lens = []
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
            traj_lens.append(len(act_seq))
            print(f"Collected trajectory {ep + 1}/{args.num_episodes} (len={len(act_seq)})")

        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        demos_path = Path(args.output_dir) / "expert_trajectories.pkl"
        with open(demos_path, "wb") as f:
            pickle.dump(trajectories, f)
        print(f"Saved demonstrations: {demos_path}")
    finally:
        env.close()
        mean_len = float(np.mean(traj_lens)) if "traj_lens" in locals() and traj_lens else 0.0
        _finish_wandb(
            run,
            demos_path=str(Path(args.output_dir) / "expert_trajectories.pkl"),
            num_trajectories=len(traj_lens) if "traj_lens" in locals() else 0,
            mean_traj_len=mean_len,
        )


def train_airl(args):
    PPO, Monitor, DummyVecEnv, (AIRL, _, BasicShapedRewardNet) = _require_optional_deps()
    run, tb_log_dir = _maybe_init_wandb(args, "train_airl")
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
    try:
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
            tensorboard_log=tb_log_dir,
        )

        reward_net = BasicShapedRewardNet(venv.observation_space, venv.action_space)
        airl_trainer = AIRL(
            demonstrations=demonstrations,
            demo_batch_size=args.demo_batch_size,
            gen_replay_buffer_capacity=args.gen_replay_buffer_capacity,
            n_disc_updates_per_round=args.n_disc_updates_per_round,
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
    finally:
        _finish_wandb(run, airl_policy_path=str(Path(args.output_dir) / "airl_generator_policy.zip"))


def render_policy(args):
    PPO, _, _, _ = _require_optional_deps()
    run, _ = _maybe_init_wandb(args, "render_policy")
    cfg = DriveEnvConfig(
        map_dir=args.map_dir,
        num_maps=args.num_maps,
        episode_length=args.episode_length,
        goal_behavior=args.goal_behavior,
        action_type=args.action_type,
        dynamics_model=args.dynamics_model,
        termination_mode=args.termination_mode,
        render_mode=args.render_mode,
    )

    env = SingleAgentDriveEnv(cfg)
    model = PPO.load(args.policy_path, device=args.device)

    steps_rendered = 0
    episodes_rendered = 0
    capture_proc = None
    try:
        # Create the render window before starting ffmpeg capture.
        obs, _ = env.reset(seed=args.seed)
        env.render()
        if args.save_video_path is not None:
            capture_proc = _start_ffmpeg_capture(args)
            # Give ffmpeg a moment to attach to the X display.
            time.sleep(0.5)

        for ep in range(args.num_episodes):
            if ep > 0:
                obs, _ = env.reset(seed=args.seed + ep)
            done = False
            ep_steps = 0
            while not done and ep_steps < args.max_steps_per_episode:
                action, _ = model.predict(obs, deterministic=args.deterministic)
                obs, _, terminated, truncated, _ = env.step(action)
                env.render()
                done = terminated or truncated
                ep_steps += 1
                steps_rendered += 1

            episodes_rendered += 1
            print(f"Rendered episode {ep + 1}/{args.num_episodes} with {ep_steps} steps")
    finally:
        _stop_ffmpeg_capture(capture_proc)
        env.close()
        summary = dict(episodes_rendered=episodes_rendered, steps_rendered=steps_rendered)
        if args.save_video_path is not None:
            summary["render_video_path"] = str(Path(args.save_video_path))
        _finish_wandb(run, **summary)


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
        p.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
        p.add_argument("--wandb-project", type=str, default="pufferdrive-irl")
        p.add_argument("--wandb-entity", type=str, default=None)
        p.add_argument("--wandb-group", type=str, default=None)
        p.add_argument("--wandb-run-name", type=str, default=None)
        p.add_argument("--wandb-tags", nargs="*", default=[])

    def add_ppo_flags(p):
        p.add_argument("--learning-rate", type=float, default=3e-4)
        p.add_argument("--n-steps", type=int, default=1024)
        p.add_argument("--batch-size", type=int, default=256)
        p.add_argument("--n-epochs", type=int, default=10)
        p.add_argument("--gamma", type=float, default=0.99)
        p.add_argument("--gae-lambda", type=float, default=0.95)

    # PPO flags
    p_train_expert = subparsers.add_parser("train_expert", help="Train SB3 PPO expert policy.")
    add_common_flags(p_train_expert)
    add_ppo_flags(p_train_expert)
    p_train_expert.add_argument("--total-timesteps", type=int, default=1_000_000)

    # Collect demos flags
    p_collect = subparsers.add_parser("collect_demos", help="Collect demonstrations from trained expert.")
    add_common_flags(p_collect)
    p_collect.add_argument("--expert-path", type=str, required=True)
    p_collect.add_argument("--num-episodes", type=int, default=128)

    # AIRL flags
    p_airl = subparsers.add_parser("train_airl", help="Train AIRL baseline from demonstrations.")
    add_common_flags(p_airl)
    add_ppo_flags(p_airl)
    p_airl.add_argument("--gen-replay-buffer-capacity", type=int, default=512)
    p_airl.add_argument("--n-disc-updates-per-round", type=int, default=1)
    p_airl.add_argument("--demos-path", type=str, required=True)
    p_airl.add_argument("--demo-batch-size", type=int, default=1024)
    p_airl.add_argument("--total-timesteps", type=int, default=1_000_000)
    p_airl.add_argument(
        "--allow-variable-horizon",
        default=True,
        action="store_true",
        help="Pass through to imitation AIRL for variable-length episodes.",
    )

    # Render policy flags
    p_render = subparsers.add_parser(
        "render_policy", help="Render a trained SB3 policy in PufferDrive (live Raylib window)."
    )
    add_common_flags(p_render)
    p_render.add_argument("--policy-path", type=str, required=True)
    p_render.add_argument("--num-episodes", type=int, default=3)
    p_render.add_argument("--max-steps-per-episode", type=int, default=200)
    p_render.add_argument(
        "--render-mode",
        type=str,
        default="human",
        choices=["human", "raylib"],
        help="Drive's render backend mode.",
    )
    p_render.add_argument("--deterministic", action="store_true")
    p_render.add_argument("--save-video-path", type=str, default=None)
    p_render.add_argument("--video-fps", type=int, default=30)
    p_render.add_argument(
        "--video-size",
        type=str,
        default="1280x720",
        help='X11 capture size for ffmpeg (e.g. "1280x720").',
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
    elif args.command == "render_policy":
        render_policy(args)
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
