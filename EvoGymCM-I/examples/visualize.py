import os
import sys

import warnings
import time

warnings.filterwarnings(
    "ignore",
    message=r"pkg_resources is deprecated as an API\.",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r".*render fps.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"You are trying to run PPO on the GPU.*",
    category=UserWarning,
)

import json
import argparse
import numpy as np
from typing import Optional
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.algo_utils import *
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import evogym.envs

try:
    import imageio.v2 as imageio
except Exception:  # pragma: no cover
    imageio = None


def save_video(
    frames,
    exp_name: str,
    video_label: str,
    reward_sum: float,
    fps: int = 30,
) -> Optional[str]:
    if imageio is None:
        print("imageio not available; skip video save.")
        return None
    if not frames:
        print("No frames captured; skip video save.")
        return None

    safe_label = video_label.replace(os.sep, "_")
    safe_reward = f"{reward_sum:.4f}".replace("-", "neg")
    out_dir = os.path.join("video", exp_name)
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f"{safe_label}_r{safe_reward}_{ts}.mp4")

    with imageio.get_writer(out_path, fps=fps, macro_block_size=1) as writer:
        for frame in frames:
            writer.append_data(frame)
    return out_path

def rollout(
    env_name: str,
    n_iters: int,
    model: PPO,
    body: np.ndarray,
    connections: Optional[np.ndarray] = None,
    material_scales: Optional[np.ndarray] = None,
    seed: int = 42,
    exp_name: Optional[str] = None,
    video_label: Optional[str] = None,
    fps: int = 30,
):
    # Parallel environments
    env_kwargs = {
        'body': body,
        'connections': connections,
        "render_mode": "human",
    }
    if material_scales is not None:
        env_kwargs['material_scales'] = material_scales
    vec_env = make_vec_env(env_name, n_envs=1, seed=seed, env_kwargs=env_kwargs)

    frames = []
    actions_taken = []
    
    # Rollout
    reward_sum = 0
    start_time = time.time()
    obs = vec_env.reset()
    count = 0
    while count < n_iters:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        actions_taken.append(action)
        reward_sum += reward[0]
        count += 1
        if done:
            print(f'\nTotal reward: {reward_sum:.5f}\n')
            break
    vec_env.close()
    elapsed = max(time.time() - start_time, 1e-6)

    if exp_name is not None and video_label is not None:
        if imageio is None:
            print("imageio not available; skip video save.")
            return
        record_kwargs = dict(env_kwargs)
        record_kwargs["render_mode"] = "rgb_array"
        record_env = make_vec_env(env_name, n_envs=1, seed=seed, env_kwargs=record_kwargs)
        # Match video playback speed to actual rollout speed.
        eff_fps = max(len(actions_taken) / elapsed, 1.0)
        record_fps = record_env.envs[0].metadata.get("render_fps", eff_fps)
        obs_record = record_env.reset()
        first_frame = record_env.envs[0].render()
        if first_frame is not None:
            frames.append(first_frame)
        for action in actions_taken:
            obs_record, _, _, _ = record_env.step(action)
            frame = record_env.envs[0].render()
            if frame is not None:
                frames.append(frame)
        record_env.close()

        out_path = save_video(frames, exp_name, video_label, reward_sum, fps=record_fps)
        if out_path is not None:
            print(f"Saved video: {out_path}")

def print_robot_info(
    structure: tuple,
    material_scales: Optional[np.ndarray],
    label: Optional[str] = None,
):
    body, connections = structure
    header = f"\n{label}\n" if label else "\nRobot info:\n"
    print(header, end="")
    print("Structure (grid):")
    print(body)
    if material_scales is None:
        print("Material scales: None")
    else:
        print("Material scales:")
        print(material_scales)
        print(f"Material scales min/max: {material_scales.min():.4f}/{material_scales.max():.4f}")

def safe_input(prompt: str) -> Optional[str]:
    try:
        return input(prompt)
    except (EOFError, KeyboardInterrupt):
        return None

def visualize_codesign(args, exp_name):
    global EXPERIMENT_PARENT_DIR
    gen_list = os.listdir(os.path.join(EXPERIMENT_PARENT_DIR, exp_name))

    assert args.env_name != None, (
        'Visualizing this experiment requires an environment be specified as a command line argument. Eg: --env-name "Walker-v0"'
    )

    gen_count = 0
    while gen_count < len(gen_list):
        try:
            gen_list[gen_count] = int(gen_list[gen_count].split("_")[1])
        except:
            del gen_list[gen_count]
            gen_count -= 1
        gen_count += 1

    all_robots = []
    if "cppn" in exp_name:
        for gen in gen_list:
            gen_data_path = os.path.join(EXPERIMENT_PARENT_DIR, exp_name, "generation_" + str(gen), "output.txt")
            f = open(gen_data_path, "r")
            count = 1
            for line in f:
                all_robots.append((gen, count, float(line.split()[1])))
                #(f'{count} | {line.split()[1]} (ID: {line.split()[0]})')
                count += 1
    all_robots = sorted(all_robots, key=lambda x: x[2], reverse=True)
    num_robots_to_print_cppn = 30 if len(all_robots) > 10 else len(all_robots)

    while(True):

        if len(all_robots) > 0:
            print()
        # for i in range(num_robots_to_print_cppn):
        #     print(f'gen: {all_robots[i][0]} |\t ind: {all_robots[i][1]}|\t r: {all_robots[i][2]}')

        pretty_print(sorted(gen_list))
        print()

        gen_input = safe_input("Enter generation number: ")
        if gen_input is None:
            return
        gen_number = int(gen_input)

        gen_data_for_printing = []
        gen_data = []
        gen_data_path = os.path.join(EXPERIMENT_PARENT_DIR, exp_name, "generation_" + str(gen_number), "output.txt")
        f = open(gen_data_path, "r")
        count = 1
        for line in f:
            gen_data_for_printing.append(f'{count} | {line.split()[1]} (ID: {line.split()[0]})') 
            gen_data.append((line.split()[0], line.split()[1]))
            count += 1

        print()
        pretty_print(gen_data_for_printing)
        print()

        rank_input = safe_input("Enter robot rank: ")
        if rank_input is None:
            return
        robot_ranks = parse_range(rank_input, len(gen_data))

        iters_input = safe_input("Enter num iters: ")
        if iters_input is None:
            return
        num_iters = int(iters_input)

        for robot_rank in robot_ranks:

            robot_index = gen_data[robot_rank-1][0]
            try:
                save_path_structure = os.path.join(EXPERIMENT_PARENT_DIR, exp_name, "generation_" + str(gen_number), "structure", str(robot_index) + ".npz")
                structure, material_scales = load_structure_with_material_scales(save_path_structure)
                print_robot_info(structure, material_scales, label=f"Structure/materials for rank {robot_rank} robot (index {robot_index}):")
            except:
                print(f'\nCould not load robot strucure data at {save_path_structure}.\n')
                continue

            if num_iters == 0:
                continue
            
            save_path_controller = os.path.join(EXPERIMENT_PARENT_DIR, exp_name, "generation_" + str(gen_number), "controller", f'{robot_index}.zip')
            model = PPO.load(save_path_controller)
            rollout(
                args.env_name,
                num_iters,
                model,
                structure[0],
                structure[1],
                material_scales,
                exp_name=exp_name,
                video_label=f"gen{gen_number}_rank{robot_rank}_id{robot_index}",
            )

def visualize_group_ppo(args, exp_name):

    exp_dir = os.path.join(EXPERIMENT_PARENT_DIR, exp_name)
    out_file = os.path.join(exp_dir, 'output.json')
    out = {}
    with open(out_file, 'r') as f:
        out = json.load(f)

    jobs = list(out.keys())
    jobs_p = []
    for i, job in enumerate(jobs):
        jobs_p.append(f'{job} ({i})')

    while True:
        pretty_print(jobs_p)
        print()

        job_input = safe_input("Enter job number: ")
        if job_input is None:
            return
        job_num = int(job_input)
        while(job_num < 0 or job_num >= len(jobs)):
            job_input = safe_input("Enter job number: ")
            if job_input is None:
                return
            job_num = int(job_input)

        job = jobs[job_num]
        job_data = out[job]

        robot_data = []
        robots_p = []
        for env in job_data.keys():
            for robot in job_data[env].keys():
                reward = job_data[env][robot]
                robot_data.append((env, reward, robot))
        robot_data = sorted(robot_data, reverse=True)

        for i, data in enumerate(robot_data):
            env_name, reward, robot = data
            robots_p.append(f'{env_name}, {robot}: {reward} | ({i})')
        
        pretty_print(robots_p, max_name_length=60)
        print()

        sim_input = safe_input("Enter sim number: ")
        if sim_input is None:
            return
        sim_num = int(sim_input)
        while(sim_num < 0 or sim_num >= len(robot_data)):
            sim_input = safe_input("Enter sim number: ")
            if sim_input is None:
                return
            sim_num = int(sim_input)

        env_name, reward, robot = robot_data[sim_num]

        iters_input = safe_input("Enter num iters: ")
        if iters_input is None:
            return
        num_iters = int(iters_input)
        print()

        if num_iters == 0:
            continue

        save_path_structure = os.path.join(exp_dir, job, "structure", f"{robot}_{env_name}.npz")
        structure, material_scales = load_structure_with_material_scales(save_path_structure)
        print_robot_info(structure, material_scales, label=f"Structure/materials for robot {robot} in {env_name}:")
        
        save_path_controller = os.path.join(exp_dir, job, "controller", f"{robot}_{env_name}.zip")
        model = PPO.load(save_path_controller)
        rollout(
            env_name,
            num_iters,
            model,
            structure[0],
            structure[1],
            material_scales,
            exp_name=exp_name,
            video_label=f"{job}_{robot}_{env_name}",
        )
        
def visualize_ppo(args, exp_name):

    exp_dir = os.path.join(EXPERIMENT_PARENT_DIR, exp_name)
    out_file = os.path.join(exp_dir, 'ppo_result.json')
    out = {}
    with open(out_file, 'r') as f:
        out = json.load(f)
        
    reward = out['best_reward']
    env_name = out['env_name']
    
    print(f'\nEnvironment: {env_name}\nReward: {reward}')

    while True:
        print()
        iters_input = safe_input("Enter num iters: ")
        if iters_input is None:
            return
        num_iters = int(iters_input)
        print()

        if num_iters == 0:
            continue

        save_path_structure = os.path.join(exp_dir, "structure", f"{env_name}.npz")
        structure, material_scales = load_structure_with_material_scales(save_path_structure)
        print_robot_info(structure, material_scales, label=f"Structure/materials for {env_name}:")
        
        save_path_controller = os.path.join(exp_dir, "controller", f"{env_name}.zip")
        model = PPO.load(save_path_controller)
        rollout(
            env_name,
            num_iters,
            model,
            structure[0],
            structure[1],
            material_scales,
            exp_name=exp_name,
            video_label=f"{env_name}",
        )

def load_structure_with_material_scales(path: str):
    data = np.load(path)
    if "structure" in data.files:
        structure = data["structure"]
        connections = data.get("connections")
        material_scales = data.get("material_scales")
        return (structure, connections), material_scales

    arrs = [value for _, value in data.items()]
    if len(arrs) < 2:
        raise ValueError(f"Invalid structure file: {path}")
    return (arrs[0], arrs[1]), None

EXPERIMENT_PARENT_DIR = os.path.join('saved_data')
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument(
        '--env-name',
        help='environment to train on')
    parser.add_argument(
        '--non-det',
        action='store_true',
        default=False,
        help='whether to use a non-deterministic policy')
    args = parser.parse_args()
    args.det = not args.non_det

    exp_list = os.listdir(EXPERIMENT_PARENT_DIR)
    pretty_print(exp_list)

    exp_name = safe_input("\nEnter experiment name: ")
    if exp_name is None:
        raise SystemExit(0)
    while exp_name not in exp_list:
        exp_name = safe_input("Invalid name. Try again: ")
        if exp_name is None:
            raise SystemExit(0)

    files_in_exp_dir = os.listdir(os.path.join(EXPERIMENT_PARENT_DIR, exp_name))
    
    if 'output.json' in files_in_exp_dir: # group ppo experiment
        visualize_group_ppo(args, exp_name)
    elif 'ppo_result.json' in files_in_exp_dir: # ppo experiment
        visualize_ppo(args, exp_name)
    else: # codesign experiment
        visualize_codesign(args, exp_name)