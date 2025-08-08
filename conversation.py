from typing import List, Tuple, Any

from conversation_prompting import prompt_constructor, generate_initial_question_prompts
from vllm_wrapper import LLM


def generate_conversation(llm: LLM, sampling_params, all_pairs, all_questions, agent_properties_lst) -> Tuple[
    List[str], List[str]]:
    """
    Accepts N*2, N*1, N*1
    Returns N*1
    """

    # Populate prompts
    initial_prompts = generate_initial_question_prompts(all_pairs, all_questions, agent_properties_lst)
    print(f"The prompts given to the agents were: {initial_prompts}\n\n\n\n")
    # Generate first level response for the whole batch
    responses: List[str] = [r.outputs[0].text for r in llm.generate(initial_prompts, sampling_params)]

    # Generate reply
    reply_prompts = []

    for idx, response in enumerate(responses):
        question = all_questions[idx]
        reply_prompts.append(prompt_constructor(agent_properties_lst[idx], [question, response]))

    # Get the other LLM's response
    final_responses: List[str] = [r.outputs[0].text for r in llm.generate(reply_prompts, sampling_params)]

    return responses, final_responses
