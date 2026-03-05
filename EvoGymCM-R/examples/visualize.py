import os
import sys

import json
import csv
import argparse
import numpy as np
from typing import Optional
import time
import imageio

# Ensure local package (../evogym) is imported instead of site-packages.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.algo_utils import *
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import evogym.envs
from evogym import VOXEL_TYPES
from ppo.stiffness_wrapper import StiffnessActionWrapper

def _count_actuators(body: np.ndarray) -> int:
    # 统计执行器体素数量（H_ACT + V_ACT）
    return int(
        np.sum(body == VOXEL_TYPES['H_ACT']) + np.sum(body == VOXEL_TYPES['V_ACT'])
    )

def _use_stiffness_wrapper(model: PPO, body: np.ndarray) -> bool:
    # 动作维度=执行器+体素刚度时，启用刚度动作包装器
    num_actuators = _count_actuators(body)
    num_voxels = int(body.size)
    model_action_dim = int(np.prod(model.action_space.shape))
    return model_action_dim == (num_actuators + num_voxels)

def _flatten_action(action: np.ndarray) -> np.ndarray:
    arr = np.asarray(action)
    if arr.ndim > 1:
        arr = arr[0]
    return arr.reshape(-1)

def _round_action_values(values: np.ndarray, precision: int) -> list:
    return [round(float(v), precision) for v in values]

def _print_structured_action(
    *,
    action: np.ndarray,
    use_stiffness_wrapper: bool,
    num_actuators: int,
    action_precision: int,
) -> None:
    action_flat = _flatten_action(action)
    if use_stiffness_wrapper and action_flat.size >= num_actuators:
        motion_action = action_flat[:num_actuators]
        stiffness_action = action_flat[num_actuators:]
        payload = {
            "motion": _round_action_values(motion_action, action_precision),
            "stiffness": _round_action_values(stiffness_action, action_precision),
        }
    else:
        payload = {
            "motion": _round_action_values(action_flat, action_precision),
            "stiffness": [],
        }

    print(json.dumps(payload, ensure_ascii=False))

def _save_voxel_index_csv(
    *,
    vec_env,
    body: np.ndarray,
    csv_path: str,
    voxel_row: int = 1,
    voxel_col: int = 2,
) -> None:
    """
    初始化每 step 记录文件（2 列）：motion, stiffness。
    行顺序即 step 顺序（第1行数据是 step 0）。
    默认目标体素：第二行第三个（1-based），即 0-based 的 (1, 2)。
    """
    csv_dir = os.path.dirname(csv_path)
    if csv_dir:
        os.makedirs(csv_dir, exist_ok=True)

    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(["motion", "stiffness"])

def _resolve_target_indices(
    *,
    vec_env,
    body: np.ndarray,
    voxel_row: int = 1,
    voxel_col: int = 2,
) -> tuple[Optional[int], int]:
    """
    默认目标体素：第二行第三个（1-based），即 0-based 的 (1, 2)。
    返回：motion 索引(0-based, 可能不存在) 与 stiffness 索引(0-based)。
    """
    voxel_linear = int(voxel_row * body.shape[1] + voxel_col)
    base_env = vec_env.envs[0].unwrapped
    actuator_indices = np.asarray(base_env.get_actuator_indices('robot')).reshape(-1)
    matches = np.where(actuator_indices == voxel_linear)[0]
    motion_idx = int(matches[0]) if len(matches) > 0 else None
    stiffness_idx = voxel_linear
    return motion_idx, stiffness_idx

def _append_step_voxel_csv(
    *,
    csv_path: str,
    motion_action: np.ndarray,
    stiffness_action: np.ndarray,
    motion_idx: Optional[int],
    stiffness_idx: int,
    action_precision: int,
) -> None:
    motion_val = ""
    if motion_idx is not None and 0 <= motion_idx < len(motion_action):
        motion_val = round(float(motion_action[motion_idx]), action_precision)

    stiffness_val = ""
    if 0 <= stiffness_idx < len(stiffness_action):
        stiffness_val = round(float(stiffness_action[stiffness_idx]), action_precision)

    with open(csv_path, 'a', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow([motion_val, stiffness_val])

def rollout(
    env_name: str,
    n_iters: int,
    model: PPO,
    body: np.ndarray,
    connections: Optional[np.ndarray] = None,
    seed: int = 42,
    deterministic: bool = True,
    record_video: bool = False,
    video_dir: str = "video",
    video_name: Optional[str] = None,
    exp_name: Optional[str] = None,
    reward_value: Optional[float] = None,
    one_episode: bool = False,
    log_actions: bool = False,
    action_precision: int = 4,
    index_csv_path: str = "voxel_action_index_row2_col3.csv",
):
    # Parallel environments
    env_kwargs = {
        'body': body,
        'connections': connections,
        "render_mode": "human",
    }
    if _use_stiffness_wrapper(model, body):
        # 兼容刚度扩展动作空间
        vec_env = make_vec_env(
            env_name,
            n_envs=1,
            seed=seed,
            env_kwargs=env_kwargs,
            wrapper_class=StiffnessActionWrapper,
            wrapper_kwargs={'body': body},
        )
    else:
        vec_env = make_vec_env(env_name, n_envs=1, seed=seed, env_kwargs=env_kwargs)

    use_stiffness = _use_stiffness_wrapper(model, body)
    num_actuators = _count_actuators(body)

    _save_voxel_index_csv(
        vec_env=vec_env,
        body=body,
        csv_path=index_csv_path,
    )
    target_motion_idx, target_stiffness_idx = _resolve_target_indices(
        vec_env=vec_env,
        body=body,
    )

    writer = None
    screenshot_steps = {202, 301, 400, 479}
    captured_steps = set()
    screenshot_dir = None
    screenshot_prefix = None
    if record_video:
        os.makedirs(video_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        if not video_name:
            base_name = exp_name if exp_name else env_name
            if reward_value is not None:
                reward_str = f"{reward_value:.4f}"
                base_name = f"{base_name}_reward{reward_str}"
            video_name = f"{base_name}_{timestamp}.mp4"
        video_path = os.path.join(video_dir, video_name)
        print(f"Recording video to: {video_path}")
        screenshot_prefix = os.path.splitext(video_name)[0]
        screenshot_dir = os.path.join(video_dir, f"{screenshot_prefix}_frames")
        os.makedirs(screenshot_dir, exist_ok=True)
        print(f"Saving snapshots to: {screenshot_dir}")
        base_env = vec_env.envs[0].unwrapped
        viewer = base_env.default_viewer
        fps = getattr(viewer, "_target_rps", None) or 50
        writer = imageio.get_writer(video_path, fps=fps)
    
    # Rollout
    reward_sum = 0
    obs = vec_env.reset()
    count = 0
    try:
        while count < n_iters:
            current_step = count
            action, _states = model.predict(obs, deterministic=deterministic)
            obs, reward, done, info = vec_env.step(action)
            action_flat = _flatten_action(action)
            if use_stiffness and action_flat.size >= num_actuators:
                motion_action = action_flat[:num_actuators]
                stiffness_action = action_flat[num_actuators:]
            else:
                motion_action = action_flat
                stiffness_action = np.array([], dtype=float)

            _append_step_voxel_csv(
                csv_path=index_csv_path,
                motion_action=motion_action,
                stiffness_action=stiffness_action,
                motion_idx=target_motion_idx,
                stiffness_idx=target_stiffness_idx,
                action_precision=action_precision,
            )

            # 兼容旧参数：默认关闭终端动作输出
            if log_actions:
                _print_structured_action(
                    action=action,
                    use_stiffness_wrapper=use_stiffness,
                    num_actuators=num_actuators,
                    action_precision=action_precision,
                )
            if writer is not None:
                base_env = vec_env.envs[0].unwrapped
                viewer = base_env.default_viewer
                viewer._init_viewer()
                if not viewer._has_init_img_camera:
                    viewer._init_img_camera()
                    viewer._has_init_img_camera = True
                viewer._viewer.render(viewer.img_camera, False, False, False, False)
                frame = viewer.img_camera.get_image()
                frame = np.array(frame)
                frame.resize(viewer.img_camera.get_resolution_height(), viewer.img_camera.get_resolution_width(), 3)
                writer.append_data(frame)
                if (
                    screenshot_dir is not None
                    and screenshot_prefix is not None
                    and current_step in screenshot_steps
                    and current_step not in captured_steps
                ):
                    screenshot_path = os.path.join(
                        screenshot_dir,
                        f"{screenshot_prefix}_step{current_step}.png",
                    )
                    imageio.imwrite(screenshot_path, frame)
                    captured_steps.add(current_step)
                    print(f"Saved snapshot: {screenshot_path}")
            reward_sum += reward[0]
            count += 1
            # vec_env 返回的是批量 done
            if done[0]:
                print(f'\nTotal reward: {reward_sum:.5f}\n')
                break
                if one_episode:
                    break
    finally:
        vec_env.close()
        if writer is not None:
            writer.close()

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
        for i in range(num_robots_to_print_cppn):
            print(f'gen: {all_robots[i][0]} |\t ind: {all_robots[i][1]}|\t r: {all_robots[i][2]}')

        pretty_print(sorted(gen_list))
        print()

        print("Enter generation number: ", end="")
        gen_number = int(input())

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

        print("Enter robot rank: ", end="")
        robot_ranks = parse_range(input(), len(gen_data))

        print("Enter num iters: ", end="")
        num_iters = int(input())

        for robot_rank in robot_ranks:

            robot_index = gen_data[robot_rank-1][0]
            try:
                save_path_structure = os.path.join(EXPERIMENT_PARENT_DIR, exp_name, "generation_" + str(gen_number), "structure", str(robot_index) + ".npz")
                structure_data = np.load(save_path_structure)
                structure = []
                for key, value in structure_data.items():
                    structure.append(value)
                structure = tuple(structure)
                print(f'\nStructure for rank {robot_rank} robot (index {robot_index}):\n{structure}\n')
            except:
                print(f'\nCould not load robot strucure data at {save_path_structure}.\n')
                continue

            if num_iters == 0:
                continue
            
            save_path_controller = os.path.join(EXPERIMENT_PARENT_DIR, exp_name, "generation_" + str(gen_number), "controller", f'{robot_index}.zip')
            model = PPO.load(save_path_controller)
            reward_value = float(gen_data[robot_rank - 1][1])
            rollout(
                args.env_name,
                num_iters,
                model,
                structure[0],
                structure[1],
                deterministic=args.det,
                record_video=args.record_video,
                video_dir=args.video_dir,
                video_name=args.video_name,
                exp_name=exp_name,
                reward_value=reward_value,
                one_episode=args.one_episode,
                log_actions=args.log_actions,
                action_precision=args.action_precision,
                index_csv_path=args.index_csv_path,
            )
            if args.one_episode:
                return

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

        print("Enter job number: ", end="")
        job_num = int(input())
        while(job_num < 0 or job_num >= len(jobs)):
            print("Enter job number: ", end="")
            job_num = int(input())

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

        print("Enter sim number: ", end="")
        sim_num = int(input())
        while(sim_num < 0 or sim_num >= len(robot_data)):
            print("Enter sim number: ", end="")
            sim_num = int(input())

        env_name, reward, robot = robot_data[sim_num]

        print("Enter num iters: ", end="")
        num_iters = int(input())
        print()

        if num_iters == 0:
            continue

        save_path_structure = os.path.join(exp_dir, job, "structure", f"{robot}_{env_name}.npz")
        structure_data = np.load(save_path_structure)
        structure = []
        for key, value in structure_data.items():
            structure.append(value)
        structure = tuple(structure)
        
        save_path_controller = os.path.join(exp_dir, job, "controller", f"{robot}_{env_name}.zip")
        model = PPO.load(save_path_controller)
        reward_value = float(reward)
        rollout(
            env_name,
            num_iters,
            model,
            structure[0],
            structure[1],
            deterministic=args.det,
            record_video=args.record_video,
            video_dir=args.video_dir,
            video_name=args.video_name,
            exp_name=exp_name,
            reward_value=reward_value,
            one_episode=args.one_episode,
            log_actions=args.log_actions,
            action_precision=args.action_precision,
            index_csv_path=args.index_csv_path,
        )
        if args.one_episode:
            return
        
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
        print("Enter num iters: ", end="")
        num_iters = int(input())
        print()

        if num_iters == 0:
            continue

        save_path_structure = os.path.join(exp_dir, "structure", f"{env_name}.npz")
        structure_data = np.load(save_path_structure)
        structure = []
        for key, value in structure_data.items():
            structure.append(value)
        structure = tuple(structure)
        
        save_path_controller = os.path.join(exp_dir, "controller", f"{env_name}.zip")
        model = PPO.load(save_path_controller)
        reward_value = float(reward)
        rollout(
            env_name,
            num_iters,
            model,
            structure[0],
            structure[1],
            deterministic=args.det,
            record_video=args.record_video,
            video_dir=args.video_dir,
            video_name=args.video_name,
            exp_name=exp_name,
            reward_value=reward_value,
            one_episode=args.one_episode,
            log_actions=args.log_actions,
            action_precision=args.action_precision,
            index_csv_path=args.index_csv_path,
        )
        if args.one_episode:
            return

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
    parser.add_argument(
        '--no-record-video',
        action='store_true',
        default=False,
        help='disable video recording (recording is enabled by default)')
    parser.add_argument(
        '--no-one-episode',
        action='store_true',
        default=False,
        help='disable single-episode run (single-episode is enabled by default)')
    parser.add_argument(
        '--video-dir',
        default='video',
        help='directory to save recorded videos')
    parser.add_argument(
        '--video-name',
        default=None,
        help='output video filename (default: <exp>_reward<value>_<timestamp>.mp4)')
    parser.add_argument(
        '--no-log-actions',
        action='store_true',
        default=False,
        help='disable structured per-step action logging (enabled by default)')
    parser.add_argument(
        '--action-precision',
        type=int,
        default=4,
        help='floating-point precision for action logging')
    parser.add_argument(
        '--index-csv-path',
        type=str,
        default='voxel_action_index_row2_col3.csv',
        help='csv file path for recording voxel motion/stiffness index mapping')
    args = parser.parse_args()
    args.det = not args.non_det
    args.record_video = not args.no_record_video
    args.one_episode = not args.no_one_episode
    # 按需求默认关闭终端 action 输出
    args.log_actions = False

    exp_list = os.listdir(EXPERIMENT_PARENT_DIR)
    pretty_print(exp_list)

    print("\nEnter experiment name: ", end="")
    exp_name = input()
    while exp_name not in exp_list:
        print("Invalid name. Try again:")
        exp_name = input()

    files_in_exp_dir = os.listdir(os.path.join(EXPERIMENT_PARENT_DIR, exp_name))
    
    if 'output.json' in files_in_exp_dir: # group ppo experiment
        visualize_group_ppo(args, exp_name)
    elif 'ppo_result.json' in files_in_exp_dir: # ppo experiment
        visualize_ppo(args, exp_name)
    else: # codesign experiment
        visualize_codesign(args, exp_name)