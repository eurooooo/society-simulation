from typing import List, Protocol

class BackBoneLLM(Protocol):
    def generate(self):
        pass
