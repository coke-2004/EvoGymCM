from typing import List
from copy import deepcopy

import numpy as np

from ga.eval import rollout_eval_fixed_policy
from utils.algo_utils import Structure

MATERIAL_MIN = 0.5
MATERIAL_MAX = 2.0


def propose_material(
    body: np.ndarray,
    s_best: np.ndarray,
    p_voxel: float = 0.2,
    sigma: float = 0.2,
) -> np.ndarray:
    """
    Generate a candidate material matrix from s_best.
    """
    occupied_mask = body > 0

    select_mask = (np.random.rand(*body.shape) < p_voxel) & occupied_mask
    # 如果没有选中任何voxel，则随机选中一个occupied voxel
    if select_mask.sum() == 0:
        occupied_indices = np.argwhere(occupied_mask)
        idx = occupied_indices[np.random.randint(len(occupied_indices))]
        select_mask[tuple(idx)] = True
        
    factor = np.ones_like(s_best, dtype=float)
    epsilon = np.random.normal(0.0, sigma, size=s_best.shape)
    factor[select_mask] = np.exp(epsilon[select_mask])

    s_candidate = s_best * factor
    s_candidate = np.where(
        occupied_mask,
        np.clip(s_candidate, MATERIAL_MIN, MATERIAL_MAX),
        1.0,
    )

    return s_candidate


def material_local_search(
    survivor: Structure,
    env_name: str,
    eval_seeds: List[int],
    T: int = 10,
    p_voxel: float = 0.2,
    sigma: float = 0.2,
) -> None:
    """
    Perform best-first local search on materials for one survivor.
    """
    if survivor.policy_path is None:
        raise ValueError("policy_path is required for material local search")

    # print('==============================================================')
    # print(f'Starting material local search for survivor with body:\n{survivor.body}\ncurrent material scales:\n{survivor.material_scales}\npolicy path: {survivor.policy_path}')
    s_best = deepcopy(survivor.material_scales)
    R_best = survivor.fitness

    for _ in range(T):
        s_candidate = propose_material(
            survivor.body,
            s_best,
            p_voxel=p_voxel,
            sigma=sigma,
        )

        # empty_voxel_mask = (survivor.body == 0)
        # if np.all(s_candidate[empty_voxel_mask] == 1.0):
        #     print("yes")
        # else:
        #     print("no")
        
        R_candidate = rollout_eval_fixed_policy(
            survivor.body,
            survivor.connections,
            s_candidate,
            survivor.policy_path,
            env_name,
            eval_seeds,
        )

        # print('---')
        # print(f'Candidate material scales:\n{s_candidate}, Best material scales:\n{s_best}')
        # print(f'Candidate material reward: {R_candidate}, Best reward: {R_best}')

        delta = 0.02 * abs(R_best)
        if R_candidate >= R_best + delta:
            # print(f'Found better material scales with reward original {R_best} and candidate {R_candidate}!')
            # print(f'Candidate material scales:\n{s_candidate}, Best material scales:\n{s_best}')
            s_best = s_candidate
            R_best = R_candidate


    # print('==============================================================')
    print(f'Original reward:{survivor.reward}, Final best reward: {R_best}')
    survivor.material_scales = s_best
    survivor.reward = R_best
    survivor.fitness = R_best


def batch_material_search(
    survivors: List[Structure],
    env_name: str,
    eval_seeds: List[int],
    T: int = 10,
    p_voxel: float = 0.2,
    sigma: float = 0.2,
) -> None:
    for survivor in survivors:
        material_local_search(
            survivor,
            env_name,
            eval_seeds,
            T=T,
            p_voxel=p_voxel,
            sigma=sigma,
        )
