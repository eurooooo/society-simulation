import torch
import random


def batched_cdist(locs, batch_size=10000):
    num_agents = locs.shape[0]
    all_dists = torch.zeros((num_agents, num_agents), device=locs.device)
    for i in range(0, num_agents, batch_size):
        end = min(i + batch_size, num_agents)
        all_dists[i:end] = torch.cdist(locs[i:end], locs)
    return all_dists


# Choose among k nearest neighbors at random
# Returns pairs of agents
def assign_pairs1(locs, top_k=3):
    # Get pairwise distances
    all_dists = batched_cdist(locs, 10000)
    # Set diagonal to large value so that we don't communicate with ourselves
    # Max distances is sqrt(2) within grid, anything greater is impossible
    max_val = 1000
    _ = all_dists.fill_diagonal_(max_val)

    all_pairs = []

    for agent1 in torch.randperm(locs.shape[0]):
        # Extract the distances for this agent
        cur_dists = all_dists[agent1]

        # Check if this agent already has a partner
        if torch.min(cur_dists) == max_val:
            continue

        # Get k nearest neighbors
        closest_k = torch.argsort(cur_dists)[:top_k]

        # Check if any partners in top_k already have an agent they are going to talk to
        # This is valuable when the agents remaining is less than top_k
        #   (i.e. when this for loop is near its conclusion)
        max_n = torch.cumsum(cur_dists[closest_k] < max_val, dim=0)[-1]
        closest_k = closest_k[:max_n]

        agent2 = random.choice(closest_k)

        all_dists[agent1, :] = max_val
        all_dists[agent2, :] = max_val
        all_dists[:, agent1] = max_val
        all_dists[:, agent2] = max_val

        all_pairs.append((agent1.item(), agent2.item()))

    return all_pairs

import torch
import random
import numpy as np
import faiss


def assign_pairs_faiss(locs, top_k=3):
    """
    Finds pairs of agents using a FAISS IndexFlatL2.
    Each agent i is matched with a random agent among its top_k nearest neighbors,
    as long as the neighbor is not already taken.
    """

    # Ensure locs is on CPU and in float32 for FAISS
    locs_np = locs.cpu().numpy().astype(np.float32)
    n, d = locs_np.shape

    # Build a flat (exact) L2 index in FAISS
    index = faiss.IndexFlatL2(d)
    index.add(locs_np)

    # Array to mark which agents are already paired
    taken = torch.zeros(n, dtype=torch.bool)

    all_pairs = []

    # Shuffle the agent indices so each agent gets a "fair shot"
    for agent1 in torch.randperm(n):
        if taken[agent1]:
            continue

        # Search for top_k + 1 neighbors because the search will likely include agent1 itself at distance 0
        distances, indices = index.search(locs_np[agent1].reshape(1, -1), top_k + 1)
        neighbors = indices[0]  # shape: (top_k+1,)

        # Remove ourselves (agent1) from the neighbor list
        neighbors = [idx for idx in neighbors if idx != agent1.item()]

        # Filter out neighbors already taken
        neighbors = [idx for idx in neighbors if not taken[idx]]

        if len(neighbors) == 0:
            # No valid neighbors left for this agent
            continue

        # Pick one neighbor at random
        agent2 = random.choice(neighbors)

        # Mark both agents as taken
        taken[agent1] = True
        taken[agent2] = True

        # Save the pair
        all_pairs.append((agent1.item(), agent2))

    return all_pairs



if __name__ == '__main__':
    # Example usage with 250k agents at the same location (for demonstration)
    dummy_locs = torch.tensor([[0.1, 0.1] for _ in range(500000)], dtype=torch.float32)

    pairs = assign_pairs_faiss(dummy_locs, top_k=3)
    print("Number of pairs:", len(pairs))
    if pairs:
        print("Example pair:", pairs[0])

