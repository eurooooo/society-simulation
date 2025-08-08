#!/usr/bin/env python3
from visualize_simulation import generate_gif
import os
import torch
import argparse


# walk through the `simulations` directory
# Generate `TLI.gif` if not exist
# load all_locs.pt and bool.pt
def generate_visualizations(simulations_dir="simulations"):
    for root, dirs, files in os.walk(simulations_dir):
        generate_visualization_for_subdir(root)

def generate_visualization_for_subdir(subdir):
    all_locs_path = os.path.join(subdir, "all_locs.pt")
    bool_path = os.path.join(subdir, "bool.pt")
    gif_path = os.path.join(subdir, "TLI.gif")
    metrics_path = os.path.join(subdir, "metrics.pt")

    if os.path.exists(all_locs_path) and os.path.exists(bool_path):
        if not os.path.exists(gif_path):
            all_locs = torch.load(all_locs_path)
            all_bools = torch.load(bool_path)
            metrics = torch.load(metrics_path) if os.path.exists(metrics_path) else None
            generate_gif(all_locs, all_bools, subdir, metrics)
            print(f"Generated {gif_path}")
    else:
        print(f"Required files not found in {subdir}")



def main():
    parser = argparse.ArgumentParser(description="Generate GIFs for simulation directories")
    parser.add_argument('--all', action='store_true', help="Generate GIFs for all simulation sub-directories")
    parser.add_argument('--subdir', type=str, help="Generate GIF for a specific simulation sub-directory")
    parser.add_argument('--simulations-dir', type=str, default="simulations", help="Path to the simulations directory")
    args = parser.parse_args()

    if args.all:
        generate_visualizations(args.simulations_dir)
    elif args.subdir:
        generate_visualization_for_subdir(args.subdir)
    else:
        print("Please provide either --all or --subdir flag")

if __name__ == "__main__":
    main()

# `./generate_vizualizaitons.py --all`
