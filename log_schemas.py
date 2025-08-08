from dataclasses import dataclass
import random
from typing import List, Literal, Generator, Tuple, Union

from vllm import SamplingParams

@dataclass
class StaticAgentProperty2:
    age: int
    gender: str
    location: str
    urbanicity: str
    ethnicity: str
    education: str

    # --------------------------------------------------------------------
    # State population data:
    # --------------------------------------------------------------------
    _STATES_DATA = [
        ("California", 39431263),
        ("Texas", 31290831),
        ("Florida", 23372215),
        ("New York", 19867248),
        ("Pennsylvania", 13078751),
        ("Illinois", 12710158),
        ("Ohio", 11883304),
        ("Georgia", 11180878),
        ("North Carolina", 11046024),
        ("Michigan", 10140459),
        ("New Jersey", 9500851),
        ("Virginia", 8811195),
        ("Washington", 7958180),
        ("Arizona", 7582384),
        ("Tennessee", 7227750),
        ("Massachusetts", 7136171),
        ("Indiana", 6924275),
        ("Missouri", 6263220),
        ("Maryland", 6245466),
        ("Wisconsin", 5960975),
        ("Colorado", 5957493),
        ("Minnesota", 5793151),
        ("South Carolina", 5478831),
        ("Alabama", 5157699),
        ("Louisiana", 4597740),
        ("Kentucky", 4588372),
        ("Oregon", 4272371),
        ("Oklahoma", 4095393),
        ("Connecticut", 3675069),
        ("Utah", 3503613),
        ("Iowa", 3267467),
        ("Nevada", 3241488),
        ("Arkansas", 3203295),
        ("Mississippi", 2970606),
        ("Kansas", 2943045),
        ("New Mexico", 2130256),
        ("Nebraska", 2005465),
        ("Idaho", 2001619),
        ("West Virginia", 1769979),
        ("Hawaii", 1446146),
        ("Maine", 1409032),
        ("New Hampshire", 1405012),
        ("Montana", 1137233),
        ("Rhode Island", 1112308),
        ("Delaware", 1051917),
        ("South Dakota", 924669),
        ("North Dakota", 796568),
        ("Alaska", 740133),
        ("Vermont", 702250),
        ("Wyoming", 648493),
    ]
    _TOTAL_POP = sum(pop for _, pop in _STATES_DATA)
    _LOCATION_VALUES = [state for (state, _) in _STATES_DATA]
    _LOCATION_WEIGHTS = [pop / _TOTAL_POP for (_, pop) in _STATES_DATA]

    # --------------------------------------------------------------------
    # The rest of the distribution data follows from the real-world info:
    # (approximate, bucketed for simplicity)
    # --------------------------------------------------------------------
    ATTRIBUTES = {
        "age": {
            "values": [16, 28, 40, 50, 65, 80],
            "weights": [0.16, 0.17, 0.16, 0.15, 0.16, 0.20]
        },
        "gender": {
            "values": ["male", "female"],
            "weights": [0.495, 0.505]
        },
        "ethnicity": {
            "values": ["White", "Hispanic", "Black", "Native American", "Asian"],
            "weights": [0.616, 0.187, 0.121, 0.010, 0.059]  
            # adjusted from your notes: White (61.6%), Hispanic (18.7%), Black (12.1%), 
            # Asian (5.9%), Native American (1.0%) => normalized to sum ~ 1.0
        },
        "urbanicity": {
            # Pew data: Urban (31%), Suburban (55%), Rural (14%)
            # We’ve carved out “Exurban” from Suburban for demonstration
            "values": ["Urban", "Suburban", "Exurban", "Rural"],
            "weights": [0.31, 0.50, 0.05, 0.14]
        },
        "education": {
            # Breaking out “Bachelor’s or higher” (33%) into College vs. Postgrad
            "values": [
                "Not High School",
                "High School",
                "Associate’s",
                "Some College",
                "College",
                "Postgraduate"
            ],
            "weights": [0.10, 0.29, 0.10, 0.18, 0.20, 0.13]
        },
        # Use the computed values/weights for states
        "location": {
            "values": _LOCATION_VALUES,
            "weights": _LOCATION_WEIGHTS
        },
    }

    SYSTEM_PROMPT_TEMPLATE = (
        "You are now adopting the persona of a {age}-year-old {gender} from {location}, "
        "a {urbanicity} area. You identify as {ethnicity}, and your highest level of formal "
        "education is {education}. Whenever you respond to prompts or questions, you should "
        "maintain consistency with these background details and viewpoints, grounding your "
        "answers in the lived experience and perspective of this hypothetical individual."
    )
    USER_PROMPT = "Tell me about yourself."

    @classmethod
    def _random_choice(cls, attr_name: str):
        """
        Helper method that picks a random value from the named attribute spec.
        """
        spec = cls.ATTRIBUTES[attr_name]
        return random.choices(spec["values"], weights=spec["weights"], k=1)[0]

    @classmethod
    def random_combination_gen(cls) -> Generator[Tuple[int, str, str, str, str, str], None, None]:
        """
        Infinite generator that yields a random combination of
        (age, gender, location, urbanicity, ethnicity, education),
        approximating the real-world U.S. distributions (2020–2021).
        """
        while True:
            combo = (
                cls._random_choice("age"),
                cls._random_choice("gender"),
                cls._random_choice("location"),
                cls._random_choice("urbanicity"),
                cls._random_choice("ethnicity"),
                cls._random_choice("education")
            )
            yield combo

    @classmethod
    def get_sys_prompt_template(cls):
        return cls.SYSTEM_PROMPT_TEMPLATE

    @classmethod
    def random_agent_generator(cls) -> Generator["AgentProperty2", None, None]:
        """
        Infinite generator that yields a new AgentProperty2 object
        with randomly selected attributes each time it is called.
        """
        position_counter = 0  # To keep track of generated agents (optional)
        gen = cls.random_combination_gen()

        while True:
            # Get a new random combination
            age, gender, location, urbanicity, ethnicity, education = next(gen)

            # Create a new AgentProperty2 instance
            agent = cls(
                age=age,
                gender=gender,
                location=location,
                urbanicity=urbanicity,
                ethnicity=ethnicity,
                education=education,
            )

            position_counter += 1  # Increment position for uniqueness
            yield agent  # Yield the AgentProperty2 instance

    def get_sys_prompt(self):
        return self.SYSTEM_PROMPT_TEMPLATE.format(
            age=self.age,
            gender=self.gender,
            location=self.location,
            urbanicity=self.urbanicity,
            ethnicity=self.ethnicity,
            education=self.education
        )


@dataclass
class SimulationParameters:
    num_agents: int
    step_sz: float
    num_iterations: int
    topk: int
    model: str
    sampling_params: SamplingParams
    scenario: str
    starter_questions: List[str]
    judge_prompt: str
    system_instructions: List[str]  # All the possible system prompts


@dataclass
class SimulationInitConditions:
    agent_static_properties_lst: List[StaticAgentProperty2]
    questionnaire_questions: List[str]

@dataclass
class AgentProperty:
    position: (int, int)
    is_bool: bool

    def gen_sys_inst(self):
        return ""


@dataclass
class AgentLog:
    id: int
    iteration_idx: int
    position: (int, int)
    questionnaire_responses: List[str]
    latent_attributes: List[Union[float, int]]


@dataclass
class ConversationLog:
    id: str
    conv_pair: (int, int)
    iteration_idx: int
    question: str
    reply: str
    final_response: str
    agreement_score: Literal[-1, 0, 1]


@dataclass
class MetricLog:
    iteration_idx: int
    metric_name: List[str]
    metric_scores: List[float]


# 1,16,male,New York,Urban,White,Not High School,Llama3.2-2b
if __name__ == '__main__':
    comb_generator = StaticAgentProperty2.random_combination_gen()
    all_combinations = [next(comb_generator) for i in range(2)]
    print(all_combinations)

