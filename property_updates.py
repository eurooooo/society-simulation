from typing import List

from judge_prompting import construct_judge_prompt


def llm_judge(llm, sampling_params, final_response: List[str], all_questions, all_replies) -> List[str]:
    # Judge the agreement between the two agents
    all_judge_prompts = []
    all_final_replies = []
    for idx, f_response in enumerate(final_response):
        question = all_questions[idx]
        reply_text = all_replies[idx]
        final_reply = f_response
        all_final_replies.append(final_reply)
        all_judge_prompts.append(construct_judge_prompt(question, reply_text, final_reply))

    grade = [r.outputs[0].text for r in llm.generate(all_judge_prompts, sampling_params)]
    with open("./grade_log.txt", "a") as f:
        f.write(f"grade is {grade} for the conversation: \n"
                f"{all_judge_prompts}\n\n\n")
    return grade


def update_properties(llm, sampling_params, final_response, all_questions, all_replies, all_pairs, agent_properties_lst,
                      agents_loc, step_sz, cur_agreements):
    grade = llm_judge(llm, sampling_params, final_response, all_questions, all_replies)

    # Update the current positions of the agents
    for idx, g in enumerate(grade):
        idx_0 = all_pairs[idx][0]
        idx_1 = all_pairs[idx][1]

        try:
            g_val = int(g)
            cur_agreements.append(g_val)
        except:
            print("non-integer judge output")
            # Very rarely happens, but assume average agreement if reward not shown
            cur_agreements.append(0)
            continue

        loc_1 = agents_loc[idx_0]
        loc_2 = agents_loc[idx_1]
        diff_vec = loc_1 - loc_2

        # Don't move if g_val == 0
        if g_val == -1:
            # Move Apart by step_sz
            agents_loc[idx_1] -= diff_vec * step_sz
            agents_loc[idx_0] += diff_vec * step_sz
        elif g_val == 1:
            # Move Closer by step_sz
            agents_loc[idx_1] += diff_vec * step_sz
            agents_loc[idx_0] -= diff_vec * step_sz

    # Adjust for agents that have went off the grid [0, 1] x [0, 1]
    agents_loc[agents_loc > 1] = 1
    agents_loc[agents_loc < 0] = 0

    return agents_loc
