
def construct_judge_prompt(question, reply, final_reply):
    llm_judge_sys = "Given an initial question and response from two political agents, score how much the agents agree with one another with values of -1, 0 or 1, where 1 means they agree and -1 means they disagree. Respond with an integer number only. Your response should contain no words, only a number, please."

    user_example = "Question: Are taxes too high?\n"
    user_example += "Agent 1: I believe that the wealthy should pay a fair share of taxes to support essential public services like universal healthcare, education, and infrastructure, which benefit everyone, not just the top 1%. The current tax code is often more beneficial to corporations and the ultra-wealthy than to the middle and lower classes, and it's time for a more progressive tax system.\n"
    user_example += "Agent 2: I think the tax burden is too high for individuals and businesses, and we need to simplify the tax code and reduce the number of tax brackets, so people can keep more of their hard-earned money and invest in their communities."

    user_example2 = "Question: What is your solution to the deficit?\n"
    user_example2 += "Agent 1: Cutting the welfare state.\n"
    user_example2 += "Agent 2: I agree, cutting the welfare state will decrease the deficit/"

    user_prompt = "Question: " + question + "\n"
    user_prompt += "Agent 1: " + reply + "\n"
    user_prompt += "Agent 2: " + final_reply + "\n"

    prompt = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
    prompt += llm_judge_sys + "\n"
    prompt += "<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
    prompt += user_example
    prompt += "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
    prompt += "-1"
    prompt += "<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
    prompt += user_example2
    prompt += "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
    prompt += "1"
    prompt += "<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
    prompt += user_prompt
    prompt += "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"

    return prompt
