from agent_prompting import get_sys_prompt

def prompt_constructor(agent_property, previous_conversation):
    prompt = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n" + get_sys_prompt(agent_property) + "\n"
    is_user = len(previous_conversation) % 2 == 0

    for idx, text in enumerate(previous_conversation):
        role = "user" if is_user else "assistant"
        prompt += f"<|eot_id|><|start_header_id|>{role}<|end_header_id|>\n{text}\n"
        is_user = not is_user

    prompt += "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    return prompt

def generate_initial_question_prompts(all_pairs, all_questions, agent_properties_lst):
    """
    Accepts N*2, N*1, N*1
    Returns N*1
    """
    initial_prompts = []
    for idx, pair in enumerate(all_pairs):
        question = all_questions[idx]
        agent_property = agent_properties_lst[pair[0]]
        initial_prompts.append(prompt_constructor(agent_property, [question]))
    return initial_prompts
