from functools import partial

import torch
import matplotlib.pyplot as plt
import os
import time
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.gridspec import GridSpec
import numpy as np


def update(t_idx, all_locs, all_bools):
    plt.clf()
    plt.xlim(-.1, 1.1)
    plt.ylim(-.1, 1.1)
    plt.scatter(all_locs[t_idx, all_bools, 0], all_locs[t_idx, all_bools, 1], c='red', label='GOP')
    plt.scatter(all_locs[t_idx, ~all_bools, 0], all_locs[t_idx, ~all_bools, 1], c='blue', label='DEM')


def generate_gif(all_locs, all_bools, log_dir, metrics=None):
    if metrics is not None:
        generate_gif_with_metrics(all_locs, all_bools, log_dir, metrics)
        return
    fig = plt.figure()
    ani = FuncAnimation(fig, partial(update,all_locs=all_locs, all_bools=all_bools), frames=all_locs.shape[0], interval=50)
    ani.save(os.path.join(log_dir,"TLI.gif"), dpi=300, writer=PillowWriter(fps=100))

def generate_gif_with_metrics(all_locs, all_bools, log_dir, metrics):
    num_agents = len(all_locs[0])
    num_metrics = len(metrics)
    # Using GridSpec to create graphs
    fig = plt.figure(figsize=(10, 6))
    gs = GridSpec(
        nrows=num_metrics,
        ncols=2,
        width_ratios=[2, 1],
        height_ratios=[1 for _ in range(num_metrics)],
        figure=fig
    )

    # Setting up the plot canvas
    # main simulation plot
    ax_sim = fig.add_subplot(gs[:, 0])
    ax_sim.set_xlim(0, 1)
    ax_sim.set_ylim(0, 1)
    ax_sim.set_title("Simulation (scatter)")

    # We'll create two scatter objects for bool == True and bool == False
    scat_true = ax_sim.scatter([], [], c='red', label='Bool True')
    scat_false = ax_sim.scatter([], [], c='blue', label='Bool False')

    metric_axes = [fig.add_subplot(gs[i, 1]) for i in range(num_metrics)]
    # We'll store a line object for each metric so we can update it frame by frame
    lines = []
    num_iterations = len(all_locs)
    for i, ax in enumerate(metric_axes):
        ax.set_title(f"Metric {i}")
        if i==(num_metrics-1):
            ax.set_xlabel("Iteration")
        ax.set_ylim(min(metrics[i]) - 1, max(metrics[i]) + 1)
        ax.set_xlim(0, num_iterations)

        # Create an empty line for each metric
        (line,) = ax.plot([], [], marker='o', linestyle='-', linewidth=1, markersize=1)
        lines.append(line)

    plt.tight_layout()

    def update(frame):
        """
        frame: current iteration (t)
        We'll update:
          - The scatter positions in the simulation for iteration 'frame'
          - The line data for each metric up to iteration 'frame'
        """
        # ----- Update scatter -----
        locs_t = all_locs[frame]  # shape: (num_agents, 2)
        bools_arr = all_bools.numpy()  # shape: (num_agents,)
        x_pos = locs_t[:, 0].numpy()
        y_pos = locs_t[:, 1].numpy()

        # True vs. False
        scat_true.set_offsets(np.column_stack((x_pos[bools_arr], y_pos[bools_arr])))
        scat_false.set_offsets(np.column_stack((x_pos[~bools_arr], y_pos[~bools_arr])))

        # Optionally set the title to show the frame index
        ax_sim.set_title(f"Simulation with {num_agents} agents: (t={frame})")

        # ----- Update each metric line -----
        for i, line in enumerate(lines):
            x_vals = np.arange(frame + 1)  # from 0 to 'frame'
            y_vals = metrics[i][: frame + 1]  # metric i, up to 'frame'
            line.set_data(x_vals, y_vals)

        return [scat_true, scat_false] + lines

    # Create gifs
    ani = FuncAnimation(
        fig,
        update,
        frames=num_iterations,  # or range(num_iterations)
        interval=300,  # ms between frames
        blit=False  # If you want to optimize, you can try blitting
    )
    ani.save(os.path.join(log_dir, "TLI.gif"), dpi=80, writer=PillowWriter(fps=25))
    print(f"GIF saved to {log_dir}")


def update_metric_neighbor_preference_similarity(neighbor_preference_similarity, all_locs, all_bools, idx, top_k):
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
    neighbor_preference_similarity.append((torch.sum(bool_mask) / torch.numel(bool_mask)).item())

def update_metric_avg_agreement_score(agreement_score, idx, all_agree):
    agreement_score.append(np.average(all_agree[idx]))

def main():
    cur_dir = '/home/robert/society-simulation/simulations/2025-01-08 22:14:29.126402'
    full_dir = os.path.join("simulations", cur_dir)
    all_locs = torch.load(os.path.join(full_dir, "all_locs.pt"))
    all_bools = torch.load(os.path.join(full_dir, "bool.pt"))
    all_agree = torch.load(os.path.join(full_dir, "all_agree.pt"))
    top_k = 3

    neighbor_preference_similarity = []
    agreement_score = []
    metrics = [neighbor_preference_similarity, agreement_score]
    for idx in range(len(all_locs)):
        # Metric 0: neighbor preference similarity

        update_metric_neighbor_preference_similarity(neighbor_preference_similarity, all_locs, all_bools, idx, top_k)
        # Metric 1: agreement score
        update_metric_avg_agreement_score(agreement_score, idx, all_agree)
        # Metric 2: Avg Response Time

        # Metric 3: Overall Response time

    # metric_shape = (len(metrics), metrics[0].shape)

    # metrics = torch.tensor(metrics)
    generate_gif_with_metrics(all_locs, all_bools, "./", metrics)
    print("Plotted")

if __name__ == '__main__':
    main()



# # New pipeline of generating metrics side by side
# import matplotlib.pyplot as plt
# from matplotlib.gridspec import GridSpec
# import numpy as np
# from matplotlib.animation import FuncAnimation, PillowWriter
# import torch
# import os
#
# # -------------- Create dummy data and save --------------
# subdir = "./simulations/2024-12-23 16:54:25.900739"
#
# all_locs_path = os.path.join(subdir, "all_locs.pt")
# bool_path = os.path.join(subdir, "bool.pt")
# metrics_path = os.path.join(subdir, "metrics.pt")
# gif_path = os.path.join(subdir, "TLI.gif")
#
# # -------------- Load data --------------
# all_locs = torch.load(all_locs_path)  # list of length num_iterations, each: shape (num_agents, 2)
# all_bools = torch.load(bool_path)  # shape (num_agents,) booleans
# metrics = torch.load(metrics_path)  # list of 3 metrics, each metric is a list of length num_iterations
# metrics = [metrics[0], metrics[0], metrics[0]]
# # number of metrics
# num_metrics = len(metrics)
#
# # -------------- Create figure & subplots with GridSpec --------------
# fig = plt.figure(figsize=(10, 6))
# gs = GridSpec(
#     nrows=num_metrics,
#     ncols=2,
#     width_ratios=[2, 1],
#     height_ratios=[1 for _ in range(num_metrics)],
#     figure=fig
# )
#
# # -- The left plot (main simulation) spans all rows in the first column --
# ax_sim = fig.add_subplot(gs[:, 0])
# ax_sim.set_xlim(0, 1)
# ax_sim.set_ylim(0, 1)
# ax_sim.set_title("Simulation (scatter)")
#
# # We'll create two scatter objects for bool == True and bool == False
# scat_true = ax_sim.scatter([], [], c='red', label='Bool True')
# scat_false = ax_sim.scatter([], [], c='blue', label='Bool False')
# # ax_sim.legend()
#
# # -- The right plots (one for each metric) --
# metric_axes = [fig.add_subplot(gs[i, 1]) for i in range(num_metrics)]
# # We'll store a line object for each metric so we can update it frame by frame
# lines = []
# num_iterations = len(all_locs)
# for i, ax in enumerate(metric_axes):
#     ax.set_title(f"Metric {i}")
#     ax.set_xlabel("Iteration")
#     # Set some y-limits so we can see negative/positive in this dummy example
#     ax.set_ylim(min(metrics[i])-1, max(metrics[i])+1)
#     ax.set_xlim(0, num_iterations)
#
#     # Create an empty line for each metric
#     (line,) = ax.plot([], [], marker='o', linestyle='-')
#     lines.append(line)
#
# plt.tight_layout()
#
#
# # -------------- Animation update function --------------
# def update(frame):
#     """
#     frame: current iteration (t)
#     We'll update:
#       - The scatter positions in the simulation for iteration 'frame'
#       - The line data for each metric up to iteration 'frame'
#     """
#     # ----- Update scatter -----
#     locs_t = all_locs[frame]  # shape: (num_agents, 2)
#     bools_arr = all_bools.numpy()  # shape: (num_agents,)
#     x_pos = locs_t[:, 0].numpy()
#     y_pos = locs_t[:, 1].numpy()
#
#     # True vs. False
#     scat_true.set_offsets(np.column_stack((x_pos[bools_arr], y_pos[bools_arr])))
#     scat_false.set_offsets(np.column_stack((x_pos[~bools_arr], y_pos[~bools_arr])))
#
#     # Optionally set the title to show the frame index
#     ax_sim.set_title(f"Simulation (t={frame})")
#
#     # ----- Update each metric line -----
#     for i, line in enumerate(lines):
#         x_vals = np.arange(frame + 1)  # from 0 to 'frame'
#         y_vals = metrics[i][: frame + 1]  # metric i, up to 'frame'
#         line.set_data(x_vals, y_vals)
#     return
#
#
# # -------------- Create and save the animation --------------
# ani = FuncAnimation(
#     fig,
#     update,
#     frames=num_iterations,  # or range(num_iterations)
#     interval=300,  # ms between frames
#     blit=False  # If you want to optimize, you can try blitting
# )
#
# # Save the animation as a GIF
# ani.save(gif_path, dpi=80, writer=PillowWriter(fps=25))
# print(f"GIF saved to {gif_path}")