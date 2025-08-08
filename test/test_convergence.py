import torch
import os
from datetime import datetime

MOST_RECENT_DIRS = 1
TOP_K = 5
SIMULATION_DIR = "./simulations"

import argparse
def compute_value(cur_dir, top_k):
    # Load data from the specified directory
    full_dir = os.path.join("simulations", cur_dir)
    all_locs = torch.load(os.path.join(full_dir, "all_locs.pt"))
    all_bools = torch.load(os.path.join(full_dir, "bool.pt"))
    all_vals = []
    for idx in range(len(all_locs)):
        all_dists = torch.cdist(all_locs[idx], all_locs[idx])
        # Set diagonal to large value so that we don't communicate with ourselves
        max_val = torch.max(all_dists) + 1
        _ = all_dists.fill_diagonal_(max_val)
        selected_indices = torch.argsort(all_dists, dim=1)[:, :top_k]
        bool_mask = torch.zeros_like(selected_indices, dtype=torch.bool)
        for r in range(len(selected_indices)):
            for c in range(len(selected_indices[0])):
                bool_mask[r, c] = all_bools[selected_indices[r, c]]
        for k in range(top_k):
            bool_mask[:, k] = bool_mask[:, k] == all_bools
        all_vals.append((torch.sum(bool_mask) / torch.numel(bool_mask)).item())
    all_vals = torch.tensor(all_vals)
    return all_vals[-1] > 0.7

def get_most_recent_simulation_dirs(recent_n_dirs):
    simulations_dir = SIMULATION_DIR

    # Filter directories with valid timestamp format and parse them
    directories = [d for d in os.listdir(simulations_dir) if os.path.isdir(os.path.join(simulations_dir, d))]
    valid_directories = []
    for d in directories:
        try:
            # Attempt to parse the timestamp
            datetime.strptime(d, "%Y-%m-%d %H:%M:%S.%f")
            valid_directories.append(d)
        except ValueError:
            # Skip invalid directories
            pass

    # Sort directories by their timestamp
    valid_directories.sort(key=lambda d: datetime.strptime(d, "%Y-%m-%d %H:%M:%S.%f"))

    # Get the most recent experiment directory
    most_recent_directory = valid_directories[-recent_n_dirs:] if valid_directories else None
    return most_recent_directory

def test_simulation_converge():
    assert os.path.isdir(SIMULATION_DIR), "`simulation` directory doesn't exist in the working directory"
    # Get the most recent simulation result
    # Hard coding the value

    # # Load data from the specified directory
    experiment_dirs = get_most_recent_simulation_dirs(MOST_RECENT_DIRS)
    assert experiment_dirs is not None, "No simulation found"
    for experiment_dir in experiment_dirs:
        full_dir = os.path.join(SIMULATION_DIR, experiment_dir)
        assert os.path.isdir(full_dir), f"Simulation directory {full_dir} does not exist"

        all_locs = torch.load(os.path.join(full_dir, "all_locs.pt"))
        all_bools = torch.load(os.path.join(full_dir, "bool.pt"))
        all_vals = []
        for idx in range(len(all_locs)):
            all_dists = torch.cdist(all_locs[idx], all_locs[idx])
            # Set diagonal to large value so that we don't communicate with ourselves
            max_val = torch.max(all_dists) + 1
            _ = all_dists.fill_diagonal_(max_val)
            selected_indices = torch.argsort(all_dists, dim=1)[:, :TOP_K]
            bool_mask = torch.zeros_like(selected_indices, dtype=torch.bool)
            for r in range(len(selected_indices)):
                for c in range(len(selected_indices[0])):
                    bool_mask[r, c] = all_bools[selected_indices[r, c]]
            for k in range(TOP_K):
                bool_mask[:, k] = bool_mask[:, k] == all_bools
            all_vals.append((torch.sum(bool_mask) / torch.numel(bool_mask)).item())
        all_vals = torch.tensor(all_vals)
        assert all_vals[-1] > 0.7, "Simulation not converging"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute values from simulation data.")
    parser.add_argument("--cur_dir", type=str, required=True, help="Directory containing simulation data.")
    parser.add_argument("--top_k", type=int, default=3, help="Number of nearest neighbors to consider.")
    args = parser.parse_args()
    result = compute_value(args.cur_dir, args.top_k)
    print(f"Result: {result}")