from pydantic import BaseModel
from .. import SimulationMetric, Agent, ConversationSelector
from typing import List

class SimulationStep(BaseModel):
    agent_lst = List[Agent]
    simulation_step: int
    conversation_selector: ConversationSelector
    metric: SimulationMetric

