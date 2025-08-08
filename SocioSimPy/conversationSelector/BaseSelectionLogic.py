from typing import List, Protocol
from ..agents import Agent
from pydantic import BaseModel
from abc import abstractmethod
import torch
import random

# Define the SelectionLogic interface

class SelectionLogic(BaseModel):
    @abstractmethod
    def group_agents(self, agents: List[Agent]) -> List[List[Agent]]:
        """
        Groups a list of agents into a list of lists of agents based on some criteria.

        :param agents: List of Agent objects to be grouped.
        :return: List of lists of Agent objects.
        """
        pass

class PairWiseTopKConversation(SelectionLogic):
    top_k: int = 5

    def group_agents(self, agents: List[Agent]) -> List[List[Agent]]:
        # If no agents or only one agent, no valid pairs
        if len(agents) < 2:
            return []

        # 1) Gather all agent locations into a single tensor.
        #    Make sure Agent.location is something like (x, y) and is accessible.
        locs = torch.stack([torch.tensor(agent.agent_properties.loc, dtype=torch.float)
                            for agent in agents])

        # 2) Compute pairwise distances
        all_dists = torch.cdist(locs, locs)

        # 3) Mask out diagonal so agents don't pair with themselves.
        #    Use a large value to represent "unavailable."
        max_val = 1e6
        all_dists.fill_diagonal_(max_val)

        all_pairs = []

        # 4) Randomize the order of agents to avoid bias
        for agent1_idx in torch.randperm(len(agents)):
            # Current distances for this agent to every other agent
            cur_dists = all_dists[agent1_idx]

            # If the minimum distance is max_val, it means agent1_idx is already 'used'
            if torch.min(cur_dists) == max_val:
                continue

            # 5) Identify top_k nearest neighbors
            sorted_indices = torch.argsort(cur_dists)
            closest_k = sorted_indices[:self.top_k]

            # If fewer than top_k remain, trim to the actual number
            # of neighbors whose distance is still < max_val
            valid_mask = cur_dists[closest_k] < max_val
            closest_k = closest_k[valid_mask]

            if len(closest_k) == 0:
                # No partners left who are valid
                continue

            # 6) Randomly pick one partner from among the valid top_k
            agent2_idx = random.choice(closest_k)

            # Mark both chosen agents (agent1_idx and agent2_idx) as used
            all_dists[agent1_idx, :] = max_val
            all_dists[agent2_idx, :] = max_val
            all_dists[:, agent1_idx] = max_val
            all_dists[:, agent2_idx] = max_val

            # 7) Collect the pair of actual Agent objects
            all_pairs.append([agents[agent1_idx], agents[agent2_idx]])

        return all_pairs
