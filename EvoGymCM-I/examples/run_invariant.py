import random
import numpy as np
import argparse
import os
import sys
import time

# Ensure CPU is respected when --device cpu is passed
for i, arg in enumerate(sys.argv):
    if arg == "--device" and i + 1 < len(sys.argv) and sys.argv[i + 1].lower() == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        break
    if arg.startswith("--device=") and arg.split("=", 1)[1].lower() == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        break

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
examples_root = os.path.abspath(os.path.dirname(__file__))
if examples_root not in sys.path:
    sys.path.insert(0, examples_root)

from ga.run import run_ga
from ppo.args import add_ppo_args

if __name__ == "__main__":
    seed = 4
    random.seed(seed)
    np.random.seed(seed)
    
    start_time = time.time()
    parser = argparse.ArgumentParser(description='Arguments for ga script')
    parser.add_argument('--exp-name', type=str, default='test_ga', help='Name of the experiment (default: test_ga)')
    parser.add_argument('--env-name', type=str, default='Walker-v0', help='Name of the environment (default: Walker-v0)')
    parser.add_argument('--pop-size', type=int, default=3, help='Population size (default: 3)')
    parser.add_argument('--structure_shape', type=tuple, default=(5,5), help='Shape of the structure (default: (5,5))')
    parser.add_argument('--max-evaluations', type=int, default=6, help='Maximum number of robots that will be evaluated (default: 6)')
    parser.add_argument('--num-cores', type=int, default=3, help='Number of robots to evaluate simultaneously (default: 3)')
    parser.add_argument('--finetune-timesteps', type=int, default=0, help='Fine-tune timesteps after material local search. 0 means use 20% of total-timesteps (default: 0)')
    add_ppo_args(parser)
    args = parser.parse_args()
    
    run_ga(args)
    end_time = time.time()
    print(f"Total time taken: {(end_time - start_time)/60:.2f} minutes")