import sqlite3
import json
from typing import List, Union, Tuple, Any

def fetch_all_agent_responses(
    conn: sqlite3.Connection,
    iteration_idx: int,
    agent_num: int
) -> List[Union[str, List[int], None]]:
    """
    Fetch agent questionnaire data for a specific iteration index from AgentLog.
    Returns a list `all_responses` of length `agent_num`, where each index i
    corresponds to the parsed JSON data for agent i (or "ERROR" if JSON fails).
    """
    cursor = conn.cursor()
    cursor.execute("""
        SELECT agent_id, questionnaire_r
        FROM AgentLog
        WHERE iter_idx = ?
    """, (iteration_idx,))

    all_responses: List[Union[str, List[int], None]] = [None] * agent_num

    for agent_id, raw_json in cursor.fetchall():
        try:
            all_responses[agent_id] = json.loads(raw_json)
        except json.JSONDecodeError:
            all_responses[agent_id] = "ERROR"

    return all_responses


def fetch_conversations(
    conn: sqlite3.Connection,
    iteration_idx: int
) -> List[Tuple[int, int, Any]]:
    """
    Fetch conversation rows (member1, member2, agreement_score)
    for a specific iteration index from ConversationLog.
    """
    cursor = conn.cursor()
    cursor.execute("""
        SELECT conv_member_1, conv_member_2, agreement_score
        FROM ConversationLog
        WHERE iteration_idx = ?
    """, (iteration_idx,))
    return cursor.fetchall()


def get_openness_data_for_iteration(
        db_path: str,
        iteration_idx: int,
        mask: List[int],
        agent_num: int
) -> Tuple[List[Tuple[int, int, Any]], List[List[int]], Any]:
    """
    Demonstrates how to retrieve conversation pairs from ConversationLog, then
    parse each agent's response from the single all-agent JSON in AgentLog.
    This version fetches all agent logs once, storing them in an array.

    :param db_path: Path to logs.db
    :param iteration_idx: Which iteration to fetch
    :param mask: Indices that define 'openness'
    :param agent_num: Number of agents (assumed IDs from 0..agent_num-1)
    :return: (agent raw responses), A list of tuples: (agent1_openness, agent2_openness, agreement_score)
    """

    results = []
    success_rate = 0
    attempts_rate = 0

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()

        # 1) Fetch all agents' data in a single query for the given iteration
        cursor.execute("""
            SELECT agent_id, questionnaire_r
            FROM AgentLog
            WHERE iter_idx = ?
        """, (iteration_idx,))

        # Prepare an array to store parsed questionnaire_r for each agent.
        # Initialize with None to easily identify missing records.
        all_responses: List[Union[str, List[int], None]] = [None] * agent_num

        # Fill in all_responses[agent_id] with the loaded data (or "ERROR" on JSON failure).
        for agent_id, raw_json in cursor.fetchall():
            try:
                all_responses[agent_id] = json.loads(raw_json)
            except json.JSONDecodeError:
                all_responses[agent_id] = "ERROR"

        # 2) Query conversations at this iteration
        cursor.execute("""
            SELECT conv_member_1, conv_member_2, agreement_score
            FROM ConversationLog
            WHERE iteration_idx = ?
        """, (iteration_idx,))
        conversation_rows = cursor.fetchall()

        # 3) For each conversation pair, retrieve each agent's data from the all_responses array
        for (a1, a2, agreement_score) in conversation_rows:
            attempts_rate += 1

            # If either agent is out of bounds or has no data, skip
            if a1 < 0 or a1 >= agent_num or a2 < 0 or a2 >= agent_num:
                print(f"Skipping conversation with invalid agent indices: a1={a1}, a2={a2}")
                continue

            resp1 = all_responses[a1]
            resp2 = all_responses[a2]

            # If either agent's record is missing or "ERROR", skip this conversation
            if resp1 is None or resp2 is None or resp1 == "ERROR" or resp2 == "ERROR":
                continue

            # 4) Compute "openness" from the binary mask
            a1_openness = sum(resp1[i] for i in mask)
            a2_openness = sum(resp2[i] for i in mask)

            results.append((a1_openness, a2_openness, agreement_score))
            success_rate += 1

    print(f"success {success_rate} out of attempts {attempts_rate}")
    return results, all_responses, conversation_rows


if __name__ == "__main__":
    # Example usage:
    db_path = "/home/robert/society-simulation/simulations/2025-02-13 03:40:53.388933/simulation_logs/logs.db"

    iteration = -1
    mask_indices = [0, 1, 5, 9, 15, 30]  # Example "openness" indices
    data, agent_responses_lst, conversation_lst = get_openness_data_for_iteration(db_path, iteration, mask_indices, agent_num=1000)

    for row in data:
        print(row)