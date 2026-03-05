import argparse
from typing import Optional
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from ppo.eval import eval_policy
from ppo.callback import EvalCallback


def _build_vec_env(
    body: np.ndarray,
    env_name: str,
    connections: Optional[np.ndarray],
    material_scales: Optional[np.ndarray],
    seed: int,
):
    env_kwargs = {
        'body': body,
        'connections': connections,
    }
    if material_scales is not None:
        env_kwargs['material_scales'] = material_scales
    return make_vec_env(env_name, n_envs=1, seed=seed, env_kwargs=env_kwargs)


def _build_eval_callback(
    body: np.ndarray,
    env_name: str,
    connections: Optional[np.ndarray],
    material_scales: Optional[np.ndarray],
    args: argparse.Namespace,
    model_save_dir: str,
    model_save_name: str,
):
    return EvalCallback(
        body=body,
        connections=connections,
        material_scales=material_scales,
        env_name=env_name,
        eval_every=args.eval_interval,
        n_evals=args.n_evals,
        n_envs=args.n_eval_envs,
        model_save_dir=model_save_dir,
        model_save_name=model_save_name,
        verbose=args.verbose_ppo,
    )

def run_ppo(
    args: argparse.Namespace,
    body: np.ndarray,
    env_name: str,
    model_save_dir: str,
    model_save_name: str,
    connections: Optional[np.ndarray] = None,
    material_scales: Optional[np.ndarray] = None,
    seed: int = 42,
) -> float:
    """
    Run ppo and return the best reward achieved during evaluation.
    """
    
    vec_env = _build_vec_env(body, env_name, connections, material_scales, seed)
    callback = _build_eval_callback(
        body,
        env_name,
        connections,
        material_scales,
        args,
        model_save_dir,
        model_save_name,
    )

    # Train
    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=args.verbose_ppo,
        device=args.device,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        ent_coef=args.ent_coef,
        clip_range=args.clip_range
    )
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callback,
        log_interval=args.log_interval
    )
    
    return callback.best_reward


def finetune_ppo(
    args: argparse.Namespace,
    body: np.ndarray,
    env_name: str,
    model_path: str,
    model_save_dir: str,
    model_save_name: str,
    connections: Optional[np.ndarray] = None,
    material_scales: Optional[np.ndarray] = None,
    total_timesteps: Optional[int] = None,
    baseline_reward: Optional[float] = None,
    seed: int = 42,
) -> float:
    """
    Load an existing PPO model and continue training (fine-tuning).
    Best checkpoint still overwrites <model_save_dir>/<model_save_name>.zip.
    """
    vec_env = _build_vec_env(body, env_name, connections, material_scales, seed)
    callback = _build_eval_callback(
        body,
        env_name,
        connections,
        material_scales,
        args,
        model_save_dir,
        model_save_name,
    )
    # Only overwrite checkpoint if fine-tuning beats the baseline.
    if baseline_reward is not None:
        callback.best_reward = float(baseline_reward)

    model = PPO.load(model_path, env=vec_env, device=args.device)
    model.learn(
        total_timesteps=args.total_timesteps if total_timesteps is None else total_timesteps,
        callback=callback,
        log_interval=args.log_interval,
        reset_num_timesteps=False,
    )

    return callback.best_reward