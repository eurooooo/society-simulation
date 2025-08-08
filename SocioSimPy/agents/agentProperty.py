from pydantic import BaseModel
from abc import ABC, abstractmethod

class AgentProperty(BaseModel, ABC):
    ...

