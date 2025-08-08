from typing import List
from pydantic import BaseModel
from ..metrics import AgentMetric
from ..agents import AgentProperty
from ..llmBackBone import BackBoneLLM


class Agent(BaseModel):
    llm: BackBoneLLM
    agent_properties: AgentProperty
    metric: List[AgentMetric] = None  # All the different type of metric at this step
