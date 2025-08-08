import random
from typing import List, TypedDict, Dict, Union, Tuple
from agent_prompting import get_sys_prompt

from log_schemas import StaticAgentProperty2

from calculate_latent_vec_score import questionnaire_res_to_latent_score

class ConversationSchema(TypedDict):
    conversation_topic: str
    primary_agent_response: str
    second_agent_response: str


def questionnaire_answering_prompt_constructor(agent_property: StaticAgentProperty2, previous_conversations: ConversationSchema, questionnaire_question_lst: List[str], primary_agent: bool):
    # sys_inst = get_sys_prompt(agent_property) # political debate case
    sys_inst = get_sys_prompt(agent_property) # political debate case
    conversation_topic = previous_conversations['conversation_topic']
    primary_agent_response = previous_conversations['primary_agent_response']
    second_agent_response = previous_conversations['second_agent_response']

    prompt = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n" + sys_inst
    prompt += (f"You are discussing on the topic '{conversation_topic}'. Below is the interaction between you and another agent. Answer the questionnaire base on your political standings and interactions. \n\n"
               f"=== Conversation History ===\n"
               f"")
    # Build conversation history according to whether this agent is primary or not
    if primary_agent:
        # First message => agent's output; rest => user input
        conversation_history = (
            f"Your output: {primary_agent_response}\n"          # Interpreted as agent's own output
            f"User input: {second_agent_response}\n"
        )
    else:
        # First message => user input; rest => agent output
        conversation_history = (
            f"User input: {primary_agent_response}\n"
            f"Your output: {second_agent_response}\n"
        )

    def format_questionnaire():
        formatted_questions = "\n".join(f"{idx + 1}. {question.strip()}" for idx, question in enumerate(questionnaire_question_lst))
        return formatted_questions

    example_response = [random.randint(0,1) for _ in range(len(questionnaire_question_lst))]
    prompt = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        f"{sys_inst}\n"
        f"You are discussing the topic '{conversation_topic}'.\n"
        "Below is the interaction between you and another agent.\n\n"
        "Answer the questionnaire based on your political stance and the conversation.\n"
        "Your output must follow these strict rules:\n"
        "1. The answer should be a list of binary integers (0 or 1), nothing else.\n"
        "2. Each element in the list corresponds to the questionnaire items in order.\n"
        "3. Provide no explanations, text, disclaimers, or any other output.\n"
        "4. Do not include any extra punctuation or formatting. No additional keys or words.\n\n"
        "=== Conversation History ===\n"
        f"{conversation_history}"
        "\n=== End of Conversation History ===\n\n"
        "<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
        f"{format_questionnaire()}\n\n"
        "Please respond with your answers as a single list of binary integers.\n"
        f"Example response: {example_response}\n\n"
        "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    )
    return prompt



def generate_questionnaire_answer(llm, sampling_params, all_pairs, agent_properties_lst, all_questions, all_replies,
                                  final_responses, questionnaire_question_lst: List[str]) -> Tuple[List[List[int]], List[List[Union[int, float]]]]:
    prompts = []
    agent_questionnaire_answers = ["" for _ in range(len(agent_properties_lst))]
    for i, (pair, q, r, final_r) in enumerate(zip(all_pairs, all_questions, all_replies
                                            , final_responses)):
        # q, r, final_r, questionnaire_question
        primary_agent_id, second_agent_id = pair

        # Create the conversation schema for passing to prompt constructor
        previous_conversations: ConversationSchema = {
            'conversation_topic': q,
            'primary_agent_response': r,
            'second_agent_response': final_r
        }

        # primary agent questionnaire
        primary_prompt = questionnaire_answering_prompt_constructor(
            agent_property=agent_properties_lst[i],
            previous_conversations=previous_conversations,
            questionnaire_question_lst=questionnaire_question_lst,
            primary_agent=True
        )
        prompts.append(primary_prompt)
        # secondary agent questionnaire
        secondary_prompt = questionnaire_answering_prompt_constructor(
            agent_property=agent_properties_lst[i],
            previous_conversations=previous_conversations,
            questionnaire_question_lst=questionnaire_question_lst,
            primary_agent=False
        )
        prompts.append(secondary_prompt)
    questionnaire_answers: List[str] = [r.outputs[0].text for r in llm.generate(prompts, sampling_params)]
    questionnaire_answers: List[List[int]] = parse_binary_strings(questionnaire_answers, len(questionnaire_question_lst))
    
    idx = 0
    # questionnaire_answers has length: 2*len(conversations): (conv1_prim, conv1_second, conv2_prim, conv2_second, ...)
    # Iterate through the responses, and fill the agent_questionnaire_answers list. This has length: len(agents)
    for (pair, _, _, _) in zip(all_pairs, all_questions, all_replies, final_responses):
        primary_agent_id, second_agent_id = pair
        # Primary agent's questionnaire
        agent_questionnaire_answers[primary_agent_id] = questionnaire_answers[idx]
        idx += 1
        # Secondary agent's questionnaire
        agent_questionnaire_answers[second_agent_id] = questionnaire_answers[idx]
        idx += 1

    latent_factor = questionnaire_res_to_latent_score(questionnaire_question_lst, questionnaire_answers)
    return agent_questionnaire_answers, latent_factor

def parse_binary_strings(binary_strings: List[str], list_len) -> List[List[int]]:
    parsed_lists = []

    for binary_string in binary_strings:
        # Remove any extraneous whitespace or special characters
        cleaned_string = binary_string.replace('\n', '').replace('\t', '').strip()
        try:
            # Split the string into a list of values and convert to integers
            parsed_list = [int(num.strip()) for num in cleaned_string.split(',')]

            # Validate that all elements are binary (0 or 1)
            if all(num in [0, 1] for num in parsed_list):
                parsed_lists.append(parsed_list)
            elif len(parsed_list) != list_len:
                raise ValueError("LLM fails to answer to some questionnaire questions")
            else:
                raise ValueError("Non-binary values detected")

        except Exception as e:
            print(f"Parsing error for: {binary_string}\nError: {e}")
            # If an error occurs, replace with a list of -1s of the same length
            fallback_list = [-1] * list_len
            parsed_lists.append(fallback_list)

    return parsed_lists
