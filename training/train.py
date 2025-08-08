# training/train.py
"""
Train a policy with Stable-Baselines3 (PPO) on the Trackmania env.

Usage:
    python training/train.py --logdir ./runs/exp1 --timesteps 1e6

By default this script will:
 - try to import `gym_envs.tm_env.TrackmaniaEnv`
 - if unavailable, fall back to a lightweight DummyEnv so you can test training plumbing
 - use PPO + MlpPolicy
 - write TensorBoard logs and periodic checkpoints

Prereqs:
    pip install stable-baselines3[extra] gym torch numpy
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np

try:
    # env (implement later)
    from gym_envs.tm_env import TrackmaniaEnv  # noqa
    REAL_ENV_AVAILABLE = True
except Exception:
    REAL_ENV_AVAILABLE = False

# SB3 imports
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecMonitor
import gym
from gym import spaces


# ---------------------------
# Fallback Dummy env (for testing)
# ---------------------------
class DummyTrackEnv(gym.Env):
    """
    Minimal placeholder environment:
    - observation: low-dim vector (state_dim,)
    - action: continuous vector [steer, throttle, brake]
    This allows you to debug the training loop without TM running.
    """
    metadata = {"render.modes": ["human"]}

    def __init__(self, state_dim: int = 24):
        super().__init__()
        # Example obs: speed, yaw_rate, N ray distances, etc
        self.state_dim = state_dim
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32)
        # actions: steer (-1..1), throttle (0..1), brake (0..1)
        self.action_space = spaces.Box(low=np.array([-1.0, 0.0, 0.0]), high=np.array([1.0, 1.0, 1.0]), dtype=np.float32)

        self._max_episode_steps = 1000
        self._step = 0
        self._state = np.zeros(self.state_dim, dtype=np.float32)

    def reset(self):
        self._step = 0
        # randomize initial state slightly
        self._state = np.random.normal(scale=0.01, size=(self.state_dim,)).astype(np.float32)
        return self._state

    def step(self, action):
        # Simple physics-free dynamics:
        # reward encourages non-zero throttle & forward progress in obs[0]
        steer, throttle, brake = action
        self._step += 1

        # toy next-state: add throttle to obs[0], add steer to obs[1], small noise
        self._state[0] += float(throttle) - float(brake) * 0.5
        self._state[1] += float(steer) * 0.1
        # clamp
        self._state = np.clip(self._state + np.random.normal(scale=1e-3, size=self.state_dim), -100, 100).astype(np.float32)

        # reward: progress along obs[0], penalize large steering
        reward = float(self._state[0]) - 0.1 * (float(steer) ** 2)
        done = self._step >= self._max_episode_steps
        info = {}
        return self._state, reward, done, info

    def render(self, mode="human"):
        # no-op; user can extend to save frames
        pass

    def close(self):
        pass


# ---------------------------
# Helper to create env
# ---------------------------
def make_env():
    if REAL_ENV_AVAILABLE:
        print("Using real TrackmaniaEnv from gym_envs.tm_env")
        env = TrackmaniaEnv()  # must match constructor signature of your env
    else:
        print("WARNING: TrackmaniaEnv not available; using DummyTrackEnv for testing")
        env = DummyTrackEnv(state_dim=24)

    # SB3 expects vectorized envs; wrap with DummyVecEnv
    venv = DummyVecEnv([lambda: env])

    # Optional: normalize observations/rewards for PPO stability
    venv = VecMonitor(venv)  # keeps episode lengths & rewards
    venv = VecNormalize(venv, norm_obs=True, norm_reward=False, clip_obs=10.0)
    return venv


# ---------------------------
# Training entrypoint
# ---------------------------
def main(args):
    # prepare log dir
    logdir = Path(args.logdir)
    logdir.mkdir(parents=True, exist_ok=True)

    venv = make_env()

    # model name and checkpoint callback
    model_path = logdir / "models"
    model_path.mkdir(parents=True, exist_ok=True)

    checkpoint_cb = CheckpointCallback(save_freq=args.save_freq, save_path=str(model_path), name_prefix="ppo_trackmania")
    # eval callback: try to evaluate every `eval_freq` steps (works with Dummy env too)
    eval_env = None
    if args.eval:
        # reuse the same env (could be a deterministic eval env)
        eval_env = make_env()
        eval_cb = EvalCallback(eval_env, best_model_save_path=str(model_path / "best"),
                                log_path=str(logdir / "eval"), eval_freq=args.eval_freq, deterministic=True, render=False)
    else:
        eval_cb = None

    # Create or load model
    if args.resume and (model_path / "latest.zip").exists():
        print("Resuming from latest checkpoint")
        model = PPO.load(str(model_path / "latest.zip"), env=venv, tensorboard_log=str(logdir))
    else:
        model = PPO("MlpPolicy", venv,
                    verbose=1,
                    tensorboard_log=str(logdir),
                    n_steps=args.n_steps,
                    batch_size=args.batch_size,
                    learning_rate=args.lr,
                    ent_coef=args.ent_coef,
                    gamma=args.gamma,
                    )

    # Training loop with callbacks
    cbs = [checkpoint_cb]
    if eval_cb is not None:
        cbs.append(eval_cb)

    # Run training
    total_timesteps = int(args.timesteps)
    print(f"Starting training for {total_timesteps} timesteps")
    model.learn(total_timesteps=total_timesteps, callback=cbs, tb_log_name="ppo_trackmania_run")

    # Save final model
    final_file = model_path / "final.zip"
    model.save(str(final_file))
    print(f"Model saved to {final_file}")


# ---------------------------
# CLI
# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=str, default="./runs/exp1", help="directory for tensorboard logs + models")
    parser.add_argument("--timesteps", type=float, default=1e5, help="total training timesteps")
    parser.add_argument("--save_freq", type=int, default=10000, help="checkpoint freq (in steps)")
    parser.add_argument("--eval", action="store_true", help="run eval callback during training")
    parser.add_argument("--eval_freq", type=int, default=20000, help="evaluation frequency (in steps)")
    parser.add_argument("--resume", action="store_true", help="resume from latest model if available")
    parser.add_argument("--n_steps", type=int, default=2048, help="PPO n_steps")
    parser.add_argument("--batch_size", type=int, default=64, help="PPO batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="learning rate")
    parser.add_argument("--ent_coef", type=float, default=0.0, help="entropy coefficient")
    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
    args = parser.parse_args()

    main(args)

