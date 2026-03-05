import os
import sys
import shutil
import json
import argparse
import time
import numpy as np
from stable_baselines3 import PPO

# Ensure local package (../evogym) is imported instead of site-packages.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import evogym.envs
from evogym import WorldObject

from ppo.args import add_ppo_args
from ppo.run import run_ppo
from ppo.eval import eval_policy
    
if __name__ == "__main__":    
    
    # Args
    parser = argparse.ArgumentParser(description='Arguments for PPO script')
    parser.add_argument(
        "--env-name", default="Walker-v0", type=str, help="Environment name (default: Walker-v0)"
    )
    parser.add_argument(
        "--save-dir", default="saved_data", type=str, help="Parent directory in which to save data(default: saved_data)"
    )
    parser.add_argument(
        "--exp-name", default="test_ppo", type=str, help="Name of experiment. Data saved to <save-dir/exp-name> (default: test_ppo)"
    )
    parser.add_argument(
        "--robot-path", default=os.path.join("world_data", "speed_bot.json"), type=str, help="Path to the robot json file (default: world_data/speed_bot.json)"
    )
    add_ppo_args(parser)
    # 为脚本运行设置默认参数（仍可被命令行显式覆盖）
    parser.set_defaults(device="cpu", total_timesteps=10000)
    args = parser.parse_args()

    # Resolve relative robot paths against this script directory so that
    # both `python run_ppo.py` (in examples/) and
    # `python ./examples/run_ppo.py` (in project root) work.
    if not os.path.isabs(args.robot_path):
        script_relative_robot_path = os.path.join(os.path.dirname(__file__), args.robot_path)
        if os.path.exists(script_relative_robot_path):
            args.robot_path = script_relative_robot_path
    
    # Manage dirs
    exp_dir = os.path.join(args.save_dir, args.exp_name)
    if os.path.exists(exp_dir):
        print(f'THIS EXPERIMENT ({args.exp_name}) ALREADY EXISTS')
        print("Delete and override? (y/n): ", end="")
        ans = input()
        if ans.lower() != "y":
            exit()
        shutil.rmtree(exp_dir)
    model_save_dir = os.path.join(args.save_dir, args.exp_name, "controller")
    structure_save_dir = os.path.join(args.save_dir, args.exp_name, "structure")
    save_name = f"{args.env_name}"

    # Get Robot
    robot = WorldObject.from_json(args.robot_path)
    os.makedirs(structure_save_dir, exist_ok=True)
    np.savez(os.path.join(structure_save_dir, save_name), robot.get_structure(), robot.get_connections())

    # Train
    train_start_time = time.time()
    best_reward = run_ppo(
        args=args,
        body=robot.get_structure(),
        connections=robot.get_connections(),
        env_name=args.env_name,
        model_save_dir=model_save_dir,
        model_save_name=save_name,
    )
    train_elapsed_sec = time.time() - train_start_time
    
    # Save result file
    with open(os.path.join(args.save_dir, args.exp_name, "ppo_result.json"), "w") as f:
        json.dump({
            "best_reward": best_reward,
            "env_name": args.env_name,
        }, f, indent=4)

    # Evaluate
    model = PPO.load(os.path.join(model_save_dir, save_name))
    rewards = eval_policy(
        model=model,
        body=robot.get_structure(),
        connections=robot.get_connections(),
        env_name=args.env_name,
        n_evals=1,
        n_envs=1,
        render_mode="human",
        deterministic_policy=True,
        use_stiffness_wrapper=args.use_stiffness_wrapper,
    )
    print(f"Mean reward: {np.mean(rewards)}")
    print(f"Training time: {train_elapsed_sec:.2f}s")