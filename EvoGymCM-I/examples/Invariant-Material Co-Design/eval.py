from typing import List, Optional

import numpy as np
from stable_baselines3 import PPO

from ppo.run import run_ppo
from ppo.run import finetune_ppo
from ppo.eval import eval_policy


def train_controller_and_eval(
    body: np.ndarray,
    connections: Optional[np.ndarray],
    material_scales: Optional[np.ndarray],
    args,
    env_name: str,
    save_dir: str,
    save_name: str,
) -> float:
    """
    Train controller and return fitness.
    """
    reward = run_ppo(
        args=args,
        body=body,
        env_name=env_name,
        model_save_dir=save_dir,
        model_save_name=save_name,
        connections=connections,
        material_scales=material_scales,
    )
    return reward


def rollout_eval_fixed_policy(
    body: np.ndarray,
    connections: Optional[np.ndarray],
    material_scales: Optional[np.ndarray],
    policy_path: str,
    env_name: str,
    seeds: List[int],
    deterministic_policy: bool = True,
) -> float:
    """
    Evaluate a fixed policy under paired seeds and return mean reward.
    """
    model = PPO.load(policy_path)

    rewards = []
    for seed in seeds:
        eval_rewards = eval_policy(
            model=model,
            body=body,
            env_name=env_name,
            connections=connections,
            material_scales=material_scales,
            n_evals=1,
            n_envs=1,
            deterministic_policy=deterministic_policy,
            seed=seed,
        )
        rewards.extend(eval_rewards)

    if len(rewards) == 0:
        return 0.0
    return float(np.mean(rewards))


def finetune_controller_and_eval(
    body: np.ndarray,
    connections: Optional[np.ndarray],
    material_scales: Optional[np.ndarray],
    args,
    env_name: str,
    policy_path: str,
    save_dir: str,
    save_name: str,
    total_timesteps: int,
    baseline_reward: float,
) -> float:
    """
    Fine-tune an existing controller and return fitness.
    """
    reward_during_finetune_best = finetune_ppo(
        args=args,
        body=body,
        env_name=env_name,
        model_path=policy_path,
        model_save_dir=save_dir,
        model_save_name=save_name,
        connections=connections,
        material_scales=material_scales,
        total_timesteps=total_timesteps,
        baseline_reward=baseline_reward,
    )

    print(
        f"[Finetune][{save_name}] baseline={baseline_reward:.6f}, best_during_train={reward_during_finetune_best:.6f}"
    )

    # Return best reward observed during fine-tuning as updated fitness.
    return reward_during_finetune_best