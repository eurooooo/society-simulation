import sqlite3
import os
import json
from typing import List, Tuple, Union


class SimLogger:
    def __init__(self, log_directory: str):
        os.makedirs(log_directory, exist_ok=True)
        self.db_path = os.path.join(log_directory, "logs.db")
        self._initialize_db()

    def _initialize_db(self):
        """Creates necessary tables if they do not exist."""
        connection = sqlite3.connect(self.db_path)
        cursor = connection.cursor()

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS AgentLog (
            agent_id INT,
            iter_idx INT,
            location_x INT,
            location_y INT,
            latent_attributes TEXT,
            questionnaire_r TEXT
        );
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS AgentProperties (
            agent_id INT PRIMARY KEY,
            age INT,
            gender TEXT,
            location TEXT,
            urbanicity TEXT,
            ethnicity TEXT,
            education TEXT
        );
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS ConversationLog (
            id TEXT PRIMARY KEY,
            conv_member_1 INT,
            conv_member_2 INT,
            iteration_idx INT,
            question TEXT,
            reply TEXT,
            final_response TEXT,
            agreement_score INT
        );
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS MetricLog (
            iter_idx INT,
            metric_name TEXT,
            metric TEXT
        );
        """)

        connection.commit()
        connection.close()

    def insert_agent_log(self, agent_id: int, iter_idx: int, location: Tuple[int, int],
                         questionnaire_r: List[str], latent_attributes: List[Union[int, float]]):
        connection = sqlite3.connect(self.db_path)
        cursor = connection.cursor()
        sql = """
        INSERT INTO AgentLog (agent_id, iter_idx, location_x, location_y, questionnaire_r, latent_attributes)
        VALUES (?, ?, ?, ?, ?, ?)
        """
        cursor.execute(sql, (agent_id, iter_idx, location[0], location[1],
                             json.dumps(questionnaire_r), json.dumps(latent_attributes)))
        connection.commit()
        connection.close()

    def insert_conversation_log(self, id: str, conv_pair: Tuple[int, int], iteration_idx: int,
                                question: str, reply: str, final_response: str, agreement_score: int):
        connection = sqlite3.connect(self.db_path)
        cursor = connection.cursor()
        sql = """
        INSERT INTO ConversationLog
        (id, conv_member_1, conv_member_2, iteration_idx, question, reply, final_response, agreement_score) 
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        cursor.execute(sql, (
        id, conv_pair[0], conv_pair[1], iteration_idx, question, reply, final_response, agreement_score))
        connection.commit()
        connection.close()

    def insert_metric_log(self, iter_idx: int, metric_name: List[str], metric: List[float]):
        connection = sqlite3.connect(self.db_path)
        cursor = connection.cursor()
        sql = """
        INSERT INTO MetricLog (iter_idx, metric_name, metric) 
        VALUES (?, ?, ?)
        """
        cursor.execute(sql, (iter_idx, json.dumps(metric_name), json.dumps(metric)))
        connection.commit()
        connection.close()

    def insert_agent_properties(self, agent_id, age, gender, location, urbanicity, ethnicity, education):
        connection = sqlite3.connect(self.db_path)
        cursor = connection.cursor()
        sql = """
        INSERT INTO AgentProperties (agent_id, age, gender, location, urbanicity, ethnicity, education)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        cursor.execute(sql, (agent_id, age, gender, location, urbanicity, ethnicity, education))
        connection.commit()
        connection.close()

# Example usage:
# logger = Logger("/path/to/logs")
# logger.insert_agent_log(1, 0, (10, 20), ["Yes", "No"], [0.5, 1.2])
