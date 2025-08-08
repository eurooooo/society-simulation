import sqlite3
import json


def clean_agent_log(db_path: str, response_length: int = 40, batch_size: int = 2500):
    """
    Cleans the AgentLog table in batches so that each 'questionnaire_r' field contains
    only the corresponding agent's response as a validated list of 40 integers.
    If parsing fails, the entry is replaced with "ERROR".
    """
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()

        # Count total rows
        cursor.execute("SELECT COUNT(*) FROM AgentLog")
        total_rows = cursor.fetchone()[0]
        print(f"Procesing {total_rows} rows")

        for offset in range(200*1000, total_rows, batch_size):
            print(f"Processing {offset} out of {total_rows}")
            # Fetch batch of agent logs
            cursor.execute("""
                SELECT agent_id, iter_idx, questionnaire_r
                FROM AgentLog
                LIMIT ? OFFSET ?
            """, (batch_size, offset))
            rows = cursor.fetchall()

            updates = []

            for agent_id, iter_idx, raw_response in rows:
                try:
                    parsed_data = json.loads(raw_response)
                    if isinstance(parsed_data, list) and all(isinstance(i, int) for i in parsed_data):
                        cleaned_response = raw_response  # Already valid, keep as is
                    else:
                        # Parse JSON to extract the agent's response list
                        all_responses = json.loads(raw_response)
                        if not isinstance(all_responses, list) or agent_id >= len(all_responses):
                            raise ValueError("Invalid format or agent index out of range.")

                        agent_response_str = all_responses[agent_id].strip()

                        # Split response into tokens (comma or space separated)
                        tokens = agent_response_str.split(',') if ',' in agent_response_str else agent_response_str.split()
                        parsed_list = [int(token.strip()) for token in tokens]

                        # Ensure response length and binary values
                        if len(parsed_list) != response_length or any(num not in [0, 1] for num in parsed_list):
                            raise ValueError("Response length incorrect or contains non-binary values.")

                        cleaned_response = json.dumps(parsed_list)  # Store as a JSON string
                except Exception as e:
                    print(f"Error processing agent {agent_id} at iteration {iter_idx}: {e}")
                    cleaned_response = json.dumps("ERROR")

                updates.append((cleaned_response, agent_id, iter_idx))

            # Update the database in batch
            cursor.executemany("""
                UPDATE AgentLog
                SET questionnaire_r = ?
                WHERE agent_id = ? AND iter_idx = ?
            """, updates)

            conn.commit()
            print(f"Processed {offset + batch_size} / {total_rows} records.")

        print("Database cleanup complete.")


if __name__ == "__main__":
    db_path = "/home/robert/society-simulation/simulations/2025-02-13 03:40:53.388933/simulation_logs/logs.db"
    clean_agent_log(db_path, batch_size=100)
