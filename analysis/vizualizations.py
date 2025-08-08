import json
import sqlite3

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List, Union
import matplotlib
from collections import Counter


# matplotlib.use("TkAgg")  # mac OS use this
def plot_openness_distance_scatter(distances, agreement_scores):
    """
    Plots a scatter plot showing agreement score (Y) vs.
    the distance in openness between agent1 and agent2 one iteration ahead (X).

    :param distances: List of distances in openness (iteration+1)
    :param agreement_scores: List of agreement scores for the current iteration
    """

    plt.figure(figsize=(8, 6))
    plt.scatter(distances, agreement_scores, alpha=0.6)

    plt.title("Agreement Score vs. Next Iteration Openness Distance")
    plt.xlabel("Distance between Agent1 & Agent2 Openness (iteration + 1)")
    plt.ylabel("Agreement Score (current iteration)")

    # Optionally add a regression line with seaborn:
    # sns.regplot(x=distances, y=agreement_scores, scatter=False, color='red')

    plt.tight_layout()
    plt.show()
    # If you want to save:
    plt.savefig("./distance_scatter.png")


def plot_openness_heatmap(agent1_openness, agent2_openness, agreement_scores):
    """
    Plots a heatmap showing the average agreement score for each combination
    of (agent1_openness, agent2_openness).

    :param agent1_openness: List of openness scores for agent 1
    :param agent2_openness: List of openness scores for agent 2
    :param agreement_scores: List of agreement scores
    """

    # Create a dataframe from the input lists
    data = pd.DataFrame({
        'agent1_openness': agent1_openness,
        'agent2_openness': agent2_openness,
        'agreement_score': agreement_scores
    })

    # Group by the two openness scores and compute the mean agreement score
    pivot_table = data.groupby(['agent2_openness', 'agent1_openness'])['agreement_score'].mean().unstack(fill_value=0)

    # Create a heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        pivot_table,
        annot=True,  # Set to True if you want numeric labels in each cell
        fmt=".2f",  # Format for the numeric labels
        cmap="YlGnBu"  # Choose a color map you like
    )

    plt.title("Average Agreement Score by Openness Levels")
    plt.xlabel("Agent 1 Openness")
    plt.ylabel("Agent 2 Openness")

    # Rotate x-axis labels if needed (optional):
    # plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()
    plt.savefig('./heatmap.png')


def plot_agent_openness_distributions(
        all_responses: List[Union[str, List[int]]],
        mask: List[int]
) -> None:
    """
    Given a list of agent responses (each either 'ERROR' or a list of ints),
    computes a distribution of total scores and plots a bar chart showing
    how many agents received each possible score.

    :param all_responses: Responses for each agent, either "ERROR" or a list of ints.
    :param mask: Indices that define which positions to look at in each response.
    """

    # Calculate total scores for each agent
    agent_scores = []
    for response in all_responses:
        response = json.loads(response)
        if isinstance(response, list):  # Only proceed if it's a valid list
            score = sum(response[m_idx] for m_idx in mask)
            agent_scores.append(score)

    # Count occurrences of each score
    score_counts = Counter(agent_scores)

    # Determine score range
    max_score = len(mask)
    score_range = list(range(max_score + 1))  # From 0 to max possible score
    counts = [score_counts.get(score, 0) for score in score_range]

    # Plot the bar chart
    plt.figure(figsize=(8, 5))
    plt.bar(score_range, counts, tick_label=score_range)
    plt.xlabel('Total Agent Openness Score')
    plt.ylabel('Number of Agents')
    plt.title('Distribution of Agent Openness Scores')
    plt.xticks(score_range)
    plt.tight_layout()
    plt.savefig("./score_distribution")


def fectch_agent_table(conn: sqlite3.Connection, columns=["agent_id", "questionnaire_r"], iter_idx=0):
    cursor = conn.cursor()
    column_str = ", ".join(columns)
    cursor.execute(f"""
        SELECT {column_str}
        FROM AgentLog
        WHERE iter_idx = ?
    """, (iter_idx,))
    return cursor.fetchall()

if __name__ == '__main__':
    from grab_data import get_openness_data_for_iteration
    #
    db_path = "/home/robert/society-simulation/simulations/2025-02-13 03:40:53.388933/simulation_logs/logs.db"
    # indices that are related to openness
    mask_indices = [0, 3, 6, 12, 15, 18, 23, 25]
    with sqlite3.connect(db_path) as conn:
        ret = fectch_agent_table(conn)
        all_responses = [response for _, response in ret]
        print(all_responses)
        plot_agent_openness_distributions(all_responses, mask_indices)
    #

    # # TODO: Only first 50 iterations for now
    # agent1_openness = []
    # agent2_openness = []
    # agreement_scores = []
    # distances = []
    # for iteration in range(1, 100):
    #     data_curr, agent_responses_lst, conversations_lst = get_openness_data_for_iteration(db_path, iteration,
    #                                                                                         mask_indices,
    #                                                                                         agent_num=1000)
    #     data_prev, agent_responses_lst_prev, conversations_lst_prev = get_openness_data_for_iteration(db_path,
    #                                                                                                   iteration - 1,
    #                                                                                                   mask_indices,
    #                                                                                                   agent_num=1000)
    #
    #     for i in range(len(data_curr)):
    #         a1_openness, a2_openness, agreement_score = data_curr[i]
    #         a1_open_prev, a2_oopen_prev, _ = data_prev[i]
    #         distances.append(abs(a1_open_prev - a2_oopen_prev))
    #
    #         agent1_openness.append(a1_openness)
    #         agent2_openness.append(a2_openness)
    #         agreement_scores.append(agreement_score)
    # print(f"Total of {len(agent1_openness)} conversations. ")
    # plot_openness_heatmap(agent1_openness, agent2_openness, agreement_scores)
    # plot_openness_distance_scatter(distances, agreement_scores)
