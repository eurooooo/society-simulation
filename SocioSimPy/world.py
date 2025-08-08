from pydantic import BaseModel
from typing import List
from . import BaseSelectionLogic, Metric, Conversation

class World(BaseModel):
    agent_lst: List["Agent"]
    iteration_step: int
    conversation_selector: BaseSelectionLogic
    metric: List[Metric]
    active_conversations: List[Conversation]

    def run_simulation(self):
        ...