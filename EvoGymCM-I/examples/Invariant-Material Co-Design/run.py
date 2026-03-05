import os
import csv
import numpy as np
import shutil
import random
import math
import argparse
import time
from typing import List
from copy import deepcopy

import evogym.envs
from evogym import sample_robot, hashable
import utils.mp_group as mp
from utils.algo_utils import get_percent_survival_evals, mutate, Structure
from ga.eval import train_controller_and_eval
from ga.eval import finetune_controller_and_eval
from ga.material_search import batch_material_search

def run_ga(
    args: argparse.Namespace,
):
    print()
    
    exp_name, env_name, pop_size, structure_shape, max_evaluations, num_cores = (
        args.exp_name,
        args.env_name,
        args.pop_size,
        args.structure_shape,
        args.max_evaluations,
        args.num_cores,
    )

    ### MANAGE DIRECTORIES ###
    home_path = os.path.join("saved_data", exp_name)
    start_gen = 0

    ### DEFINE TERMINATION CONDITION ###

    is_continuing = False    
    try:
        os.makedirs(home_path)
    except:
        print(f'THIS EXPERIMENT ({exp_name}) ALREADY EXISTS')
        print("Override? (y/n/c): ", end="")
        ans = input()
        if ans.lower() == "y":
            shutil.rmtree(home_path)
            print()
        elif ans.lower() == "c":
            print("Enter gen to start training on (0-indexed): ", end="")
            start_gen = int(input())
            is_continuing = True
            print()
        else:
            return

    ### STORE META-DATA ##
    csv_log_path = os.path.join("saved_data", exp_name, "eval_log.csv")
    csv_eval_index = 0
    csv_rows = [["eval_index", "reward"]]
    csv_row_pos_by_eval_index = {}
    if not is_continuing:
        temp_path = os.path.join("saved_data", exp_name, "metadata.txt")
        
        try:
            os.makedirs(os.path.join("saved_data", exp_name))
        except:
            pass

        f = open(temp_path, "w")
        f.write(f'POP_SIZE: {pop_size}\n')
        f.write(f'STRUCTURE_SHAPE: {structure_shape[0]} {structure_shape[1]}\n')
        f.write(f'MAX_EVALUATIONS: {max_evaluations}\n')
        f.close()

        with open(csv_log_path, "w", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["eval_index", "reward"])
        csv_eval_index = 0
        csv_rows = [["eval_index", "reward"]]
        csv_row_pos_by_eval_index = {}

    else:
        temp_path = os.path.join("saved_data", exp_name, "metadata.txt")
        f = open(temp_path, "r")
        count = 0
        for line in f:
            if count == 0:
                pop_size = int(line.split()[1])
            if count == 1:
                structure_shape = (int(line.split()[1]), int(line.split()[2]))
            if count == 2:
                max_evaluations = int(line.split()[1])
            count += 1

        print(f'Starting training with pop_size {pop_size}, shape ({structure_shape[0]}, {structure_shape[1]}), ' + 
            f'max evals: {max_evaluations}.')
        
        f.close()

        # Continue CSV index from existing rows if present
        if os.path.exists(csv_log_path):
            with open(csv_log_path, "r", newline="") as csv_file:
                rows = list(csv.reader(csv_file))
            csv_eval_index = max(0, len(rows) - 1)  # minus header
            if len(rows) == 0:
                csv_rows = [["eval_index", "reward"]]
            else:
                csv_rows = rows
                if csv_rows[0] != ["eval_index", "reward"]:
                    csv_rows[0] = ["eval_index", "reward"]
            csv_row_pos_by_eval_index = {}
            for pos in range(1, len(csv_rows)):
                if len(csv_rows[pos]) < 2:
                    continue
                try:
                    idx_val = int(csv_rows[pos][0])
                except ValueError:
                    continue
                csv_row_pos_by_eval_index[idx_val] = pos
        else:
            with open(csv_log_path, "w", newline="") as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(["eval_index", "reward"])
            csv_eval_index = 0
            csv_rows = [["eval_index", "reward"]]
            csv_row_pos_by_eval_index = {}

    ### GENERATE // GET INITIAL POPULATION ###
    structures: List[Structure] = []
    population_structure_hashes = {}
    num_evaluations = 0
    generation = 0
    
    #generate a population
    if not is_continuing: 
        for i in range (pop_size):
            
            temp_structure = sample_robot(structure_shape)
            while (hashable(temp_structure[0]) in population_structure_hashes):
                temp_structure = sample_robot(structure_shape)

            structures.append(Structure(*temp_structure, i))
            population_structure_hashes[hashable(temp_structure[0])] = True
            num_evaluations += 1

    #read status from file
    else:
        for g in range(start_gen+1):
            for i in range(pop_size):
                save_path_structure = os.path.join("saved_data", exp_name, "generation_" + str(g), "structure", str(i) + ".npz")
                np_data = np.load(save_path_structure)
                if "structure" in np_data.files:
                    body = np_data["structure"]
                    connections = np_data.get("connections")
                    material_scales = np_data.get("material_scales")
                    eval_index = np_data.get("eval_index")
                else:
                    arrs = [value for _, value in np_data.items()]
                    body = arrs[0]
                    connections = arrs[1] if len(arrs) > 1 else None
                    material_scales = arrs[2] if len(arrs) > 2 else None
                    eval_index = arrs[3] if len(arrs) > 3 else None

                if material_scales is None:
                    material_scales = np.ones_like(body, dtype=float)
                parsed_eval_index = None
                if eval_index is not None:
                    try:
                        parsed_eval_index = int(np.asarray(eval_index).item())
                        if parsed_eval_index < 0:
                            parsed_eval_index = None
                    except Exception:
                        parsed_eval_index = None
                population_structure_hashes[hashable(body)] = True
                # only a current structure if last gen
                if g == start_gen:
                    structures.append(Structure(body, connections, i, material_scales, parsed_eval_index))
        num_evaluations = len(list(population_structure_hashes.keys()))
        generation = start_gen


    while True:

        ### UPDATE NUM SURVIORS ###			
        percent_survival = get_percent_survival_evals(num_evaluations, max_evaluations)
        num_survivors = max(2, math.ceil(pop_size * percent_survival))


        ### MAKE GENERATION DIRECTORIES ###
        save_path_structure = os.path.join("saved_data", exp_name, "generation_" + str(generation), "structure")
        save_path_controller = os.path.join("saved_data", exp_name, "generation_" + str(generation), "controller")
        
        try:
            os.makedirs(save_path_structure)
        except:
            pass

        try:
            os.makedirs(save_path_controller)
        except:
            pass

        ### TRAIN GENERATION

        #better parallel
        group = mp.Group()
        # print(f'total-timesteps: {args.total_timesteps}')
        for structure in structures:

            if structure.is_survivor:
                save_path_controller_part = os.path.join("saved_data", exp_name, "generation_" + str(generation), "controller",
                    f"{structure.label}.zip")
                save_path_controller_part_old = os.path.join("saved_data", exp_name, "generation_" + str(generation-1), "controller",
                    f"{structure.prev_gen_label}.zip")
                
                print(f'Skipping training for {save_path_controller_part}.\n')
                try:
                    shutil.copy(save_path_controller_part_old, save_path_controller_part)
                    structure.policy_path = os.path.join(
                        "saved_data",
                        exp_name,
                        "generation_" + str(generation),
                        "controller",
                        f"{structure.label}",
                    )
                except:
                    print(f'Error coppying controller for {save_path_controller_part}.\n')
            else:
                # start_time = time.time()
                # print(f"[Train] start id={structure.label} at {time.strftime('%H:%M:%S')} ...")
                structure.policy_path = os.path.join(save_path_controller, f"{structure.label}")
                ppo_args = (
                    structure.body,
                    structure.connections,
                    structure.material_scales,
                    args,
                    env_name,
                    save_path_controller,
                    f'{structure.label}',
                )
                group.add_job(train_controller_and_eval, ppo_args, callback=structure.set_reward)
                
                # elapsed = time.time() - start_time
                # print(f"[Train] done  id={structure.label} reward={structure.reward:.4f} in {elapsed:.1f}s")

        group.run_jobs(num_cores)

        #not parallel
        #for structure in structures:
        #    ppo.run_algo(structure=(structure.body, structure.connections), termination_condition=termination_condition, saving_convention=(save_path_controller, structure.label))

        ### COMPUTE FITNESS, SORT, AND SAVE ###
        for structure in structures:
            structure.compute_fitness()

        structures = sorted(structures, key=lambda structure: structure.fitness, reverse=True)

        ### MATERIAL LOCAL SEARCH ###
        survivors = structures[:num_survivors]
        material_eval_seeds = [42]
        batch_material_search(
            survivors,
            env_name,
            material_eval_seeds,
            T=20,
            p_voxel=0.15,
            sigma=0.2,
        )

        # Fine-tune controllers on updated best materials (parallelizable)
        if len(survivors) > 0:
            finetune_timesteps = args.finetune_timesteps
            if finetune_timesteps <= 0:
                finetune_timesteps = max(1, int(args.total_timesteps * 0.01))

            finetune_group = mp.Group()
            for survivor in survivors:
                if survivor.policy_path is None:
                    continue
                finetune_args = (
                    survivor.body,
                    survivor.connections,
                    survivor.material_scales,
                    args,
                    env_name,
                    survivor.policy_path,
                    save_path_controller,
                    f"{survivor.label}",
                    finetune_timesteps,
                    survivor.fitness,
                )
                finetune_group.add_job(finetune_controller_and_eval, finetune_args, callback=survivor.set_reward)
            finetune_group.run_jobs(num_cores)

        for structure in structures:
            structure.compute_fitness()
            
        # Log rewards: new robots append rows; survivors overwrite their last row via eval_index.
        if len(structures) > 0:
            for structure in structures:
                if not structure.is_survivor:
                    structure.eval_index = csv_eval_index
                    csv_rows.append([structure.eval_index, float(structure.fitness)])
                    csv_row_pos_by_eval_index[structure.eval_index] = len(csv_rows) - 1
                    csv_eval_index += 1
                else:
                    if structure.eval_index is None:
                        continue
                    row_pos = csv_row_pos_by_eval_index.get(int(structure.eval_index))
                    if row_pos is None:
                        continue
                    csv_rows[row_pos] = [int(structure.eval_index), float(structure.fitness)]

            with open(csv_log_path, "w", newline="") as csv_file:
                writer = csv.writer(csv_file)
                writer.writerows(csv_rows)

        # Re-sort in case material search improved fitness
        structures = sorted(structures, key=lambda structure: structure.fitness, reverse=True)

        # Save final generation structure data after all updates in this generation
        for structure in structures:
            temp_path = os.path.join(save_path_structure, str(structure.label))
            np.savez(
                temp_path,
                structure=structure.body,
                connections=structure.connections,
                material_scales=structure.material_scales,
                eval_index=-1 if structure.eval_index is None else int(structure.eval_index),
            )

        #SAVE RANKING TO FILE
        temp_path = os.path.join("saved_data", exp_name, "generation_" + str(generation), "output.txt")
        f = open(temp_path, "w")

        out = ""
        for structure in structures:
            out += str(structure.label) + "\t\t" + str(structure.fitness) + "\n"
        f.write(out)
        f.close()

        print(f'FINISHED GENERATION {generation} - SEE TOP {round(percent_survival*100)} percent of DESIGNS:\n')
        # print(structures[:num_survivors])

        ### CHECK EARLY TERMINATION ###
        if num_evaluations == max_evaluations:
            print(f'Trained exactly {num_evaluations} robots')
            return

        ### CROSSOVER AND MUTATION ###
        # save the survivors
        survivors = structures[:num_survivors]

        #store survivior information to prevent retraining robots
        for i in range(num_survivors):
            structures[i].is_survivor = True
            structures[i].prev_gen_label = structures[i].label
            structures[i].label = i

        # for randomly selected survivors, produce children (w mutations)
        num_children = 0
        while num_children < (pop_size - num_survivors) and num_evaluations < max_evaluations:

            parent_index = random.sample(range(num_survivors), 1)
            child = mutate(
                deepcopy(survivors[parent_index[0]].body),
                mutation_rate = 0.1,
                num_attempts=50,
                material_scales=survivors[parent_index[0]].material_scales,
            )

            if child != None and hashable(child[0]) not in population_structure_hashes:
                
                # overwrite structures array w new child
                if len(child) >= 3:
                    structures[num_survivors + num_children] = Structure(child[0], child[1], num_survivors + num_children, child[2])
                else:
                    structures[num_survivors + num_children] = Structure(child[0], child[1], num_survivors + num_children)
                population_structure_hashes[hashable(child[0])] = True
                num_children += 1
                num_evaluations += 1

        structures = structures[:num_children+num_survivors]

        generation += 1