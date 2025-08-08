from pydantic import BaseModel
from typing import List
from abc import ABC, abstractmethod
from .. import Agent

class Conversation(BaseModel, ABC):
    conversation_agent_lst: List[Agent]
    conversation_history: str

    @abstractmethod
    def talk(self):
        raise NotImplementedError("You need to implement talk method for your custom conversation")




