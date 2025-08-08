import json
import os
import datetime
from typing import List, Tuple, Any, Union

import uuid
import torch
import random
import argparse
from log_schemas import SimulationParameters, AgentLog, ConversationLog, MetricLog
from assign_pairs import assign_pairs1

from conversation import generate_conversation
from generate_questionnaire_answer import generate_questionnaire_answer

from property_updates import update_properties
from starter_prompts import starter_prompts
from generate_vizualizaitons import generate_visualization_for_subdir
from dotenv import load_dotenv
import numpy as np

from vllm import SamplingParams
from vllm_wrapper import BatchedLLM

from log_schemas import StaticAgentProperty2
from qeustionnaire_questions import questionnaire_questions
# TODO: actually use this

from database_manager import SimLogger

QUESTIONNAIRE_QUESTIONs = questionnaire_questions

logger: SimLogger = None


def save_json(data, filename: str):
    """Utility to save `data` as a JSON file."""
    import json
    with open(filename, "w") as f:
        json.dump(data, f, indent=2, default=str)


def parse_args():
    parser = argparse.ArgumentParser(description="Simulation of multi-agent interactions")
    parser.add_argument('--num-agents', type=int, required=True, help="Number of agents in the simulation")
    parser.add_argument('--step-sz', type=float, required=True, help="Step size for agent movement")
    parser.add_argument('--num-iterations', type=int, required=True, help="Number of iterations for the simulation")
    parser.add_argument('--topk', type=int, required=True, help="Top-k nearest neighbors for pairing")
    parser.add_argument('--simulation-timestamp', default=str(datetime.datetime.now()), help="Simulation timestamp")
    parser.add_argument('--gif', action='store_true', help="Generate a gif after running the simulation.")
    parser.add_argument('--testing', action='store_true', help="Running a test without logging. ")
    return parser.parse_args()


def log_simulation_setup(num_agents, step_sz, num_iterations, topk, model, sampling_params, scenario, starter_prompts,
                         output_dir):
    from judge_prompting import construct_judge_prompt
    judge_prompt = construct_judge_prompt("{question}", "{reply}", "{final_reply}")
    system_instruction = StaticAgentProperty2.get_sys_prompt_template()
    s = SimulationParameters(num_agents, step_sz, num_iterations, topk, model, sampling_params, scenario,
                             starter_prompts,
                             judge_prompt, system_instruction)
    # Log to json file
    sim_params_dict = s.__dict__
    setup_log_path = os.path.join(output_dir, "simulation_setup.json")
    save_json(sim_params_dict, setup_log_path)


def log_simulation_init_conditions(questaionnaire_questions, static_agent_properties_lst: List[StaticAgentProperty2],
                                   output_dir):
    # TODO: Log static agent properties and other questionnaire questions. Worry about this later.
    # Log questionnaire questions
    setup_log_path = os.path.join(output_dir, "questionnaire_questions.txt")
    with open(setup_log_path, "w") as f:
        f.write("\n".join(questaionnaire_questions))

    # Log static agent properties
    for agent_id, static_agent_property in enumerate(static_agent_properties_lst):
        logger.insert_agent_properties(agent_id=agent_id, age=static_agent_property.age,
                                       gender=static_agent_property.gender,
                                       location=static_agent_property.location,
                                       urbanicity=static_agent_property.urbanicity,
                                       ethnicity=static_agent_property.ethnicity,
                                       education=static_agent_property.education)
    return


def log_agents(iter_idx, agent_locs: List[Tuple[int, int]], questionnaire_r_lst: List[str], log_dir,
               latent_attributes: List[List[Union[int, float]]]):
    # Receives a list of agents, their properties (locations, boolean values)

    for agent_id, (location, questionnaire_r, latent_attribute) in enumerate(zip(agent_locs, questionnaire_r_lst, latent_attributes)):
        AL = AgentLog(agent_id, iter_idx, location,
                      questionnaire_r, latent_attribute)
        print(f"At iteration {AL.iteration_idx}, agent {AL.id} is at position {AL.position}")
        # database log
        logger.insert_agent_log(agent_id, iter_idx, location,
                             questionnaire_r, latent_attribute)


def log_conversations(iter_idx, curr_pairs, curr_questions, curr_replies, final_responses: List[str], curr_agreements,
                      log_dir):
    # Log the list of pairs and questions
    # Log the replies and final responses
    for conv_pair, question, reply, final_response, agreement_score in zip(curr_pairs, curr_questions, curr_replies,
                                                                           final_responses,
                                                                           curr_agreements):
        conv_id = uuid.uuid4()
        CL = ConversationLog(id=str(conv_id), conv_pair=conv_pair, iteration_idx=iter_idx, question=question,
                             reply=reply, final_response=final_response, agreement_score=agreement_score)
        print(
            f"At iteration {CL.iteration_idx}, the conversation between {CL.conv_pair} was about '{CL.question}' and the reply was {CL.reply}. "
            f"\nThe final response was {CL.final_response} with an agreement score of {CL.agreement_score}")

        # database log
        logger.insert_conversation_log(CL.id, CL.conv_pair, CL.iteration_idx, CL.question, CL.reply,
                                           CL.final_response,
                                           CL.agreement_score)


def log_metrics(iter_idx, metric_names, metrics, log_dir):
    # Log the metrics
    for metric_name, metric in zip(metric_names, metrics):
        ML = MetricLog(iter_idx, metric_name, metric)
        print(f"At iteration {ML.iteration_idx}, the metric {ML.metric_name} was {ML.metric_scores}")
        logger.insert_metric_log(ML.iteration_idx, ML.metric_name, ML.metric_scores)


def main():
    # Parse CLI args
    args = parse_args()
    num_agents = args.num_agents
    step_sz = args.step_sz
    num_iterations = args.num_iterations
    topk = args.topk
    testing = args.testing
    if testing:
        top_level_dir = "./temp_logs"
    else:
        top_level_dir = os.path.join("simulations", args.simulation_timestamp)

    log_dir = os.path.join(top_level_dir, "simulation_logs")
    os.makedirs(log_dir, exist_ok=True)

    # create the logger to log agent, conversation, and simulation details
    global logger
    logger = SimLogger(log_dir)

    # Create LLM backbone
    MODEL = "meta-llama/Llama-3.1-8B-Instruct"
    SENARIO = "different demographics political debate"
    llm = BatchedLLM(model=MODEL, max_model_len=8000, enable_prefix_caching=True)
    sampling_params = SamplingParams(temperature=0.5, top_p=0.9, max_tokens=256, )

    log_simulation_setup(num_agents, step_sz, num_iterations, topk, MODEL, sampling_params, SENARIO, starter_prompts,
                         top_level_dir)

    # Initialize agents
    comb_generator = StaticAgentProperty2.random_agent_generator()
    agents_loc = torch.rand(num_agents, 2)
    agent_properties_lst = [next(comb_generator) for _ in range(num_agents)]
    log_simulation_init_conditions(QUESTIONNAIRE_QUESTIONs, agent_properties_lst, top_level_dir)

    # torch.save(agent_properties_lst, os.path.join(top_level_dir, "bool.pt"))
    all_locations = []
    all_agreements = []

    # Both of these have length of len(conversation)
    replies = []  # List of Agent replies to the conversation starter question
    final_responses = []  # The response from the other agent

    agreement_metric = []
    metrics = [agreement_metric]

    # Collect questionnaire response before simulation starts
    all_pairs = assign_pairs1(agents_loc, topk)
    questionnaire_responses, latent_vec = generate_questionnaire_answer(llm, sampling_params, all_pairs,
                                                                        agent_properties_lst,
                                                                        ["" for _ in range(len(all_pairs))],
                                                                        # No starter questions
                                                                        ["" for _ in range(len(all_pairs))],
                                                                        # No replies
                                                                        ["" for _ in range(len(all_pairs))],
                                                                        # No final response
                                                                        QUESTIONNAIRE_QUESTIONs)

    log_agents(-1, agents_loc.tolist(), questionnaire_responses,
               log_dir,
               latent_vec)

    for iter_idx in range(num_iterations):
        # Feed it just the questionnaire with no previous history

        cur_agreements = []
        if iter_idx % 10 == 0:
            print(f"Iteration: {iter_idx}")

        # Pair agents and run conversations
        all_pairs = assign_pairs1(agents_loc, topk)

        # set up questions
        all_questions = [random.choice(starter_prompts) for _ in range(len(all_pairs))]

        # Start a 1 round conversation between the two agent: len(conversation)
        all_replies, final_responses = generate_conversation(llm, sampling_params, all_pairs, all_questions,
                                                             agent_properties_lst)

        # Get the questionnaire response
        questionnaire_responses, latent_vec = generate_questionnaire_answer(llm, sampling_params, all_pairs,
                                                                            agent_properties_lst,
                                                                            all_questions, all_replies, final_responses,
                                                                            QUESTIONNAIRE_QUESTIONs)

        # update agent properties: positions, agreement score using LLM judge
        agents_loc = update_properties(llm, sampling_params, final_responses, all_questions, all_replies, all_pairs,
                                       agent_properties_lst, agents_loc, step_sz, cur_agreements)

        # Log Agents
        log_agents(iter_idx, agents_loc.tolist(),
                   questionnaire_responses, log_dir, latent_vec)
        # Log conversations
        log_conversations(iter_idx, all_pairs, all_questions, all_replies, final_responses, cur_agreements, log_dir)

        # Logging to visualize
        all_locations.append(agents_loc.detach().clone().unsqueeze(0))
        all_agreements.append(torch.tensor(cur_agreements))
        agreement_metric.append(np.average(cur_agreements))
        # Log Metrics
        log_metrics(iter_idx, ["agreement score"], metrics, log_dir)

        torch.save(torch.stack(all_agreements), os.path.join(top_level_dir, "all_agree.pt"))
        torch.save(torch.cat(all_locations), os.path.join(top_level_dir, "all_locs.pt"))
        torch.save(torch.tensor(metrics), os.path.join(top_level_dir, "metrics.pt"))

    print(f"Simulation finished and logged to {top_level_dir}")
    if args.gif:
        print(f"Generating the gif at {top_level_dir}")
        generate_visualization_for_subdir(top_level_dir)


if __name__ == "__main__":
    load_dotenv()

    main()

"""
python3 ./main.py --num-agents=500000 --step-sz=.05 --num-iterations=1 --topk=3
python3 ./main.py --num-agents=2 --step-sz=.05 --num-iterations=10 --topk=5 --gif --testing
python3 ./main.py --num-agents=10 --step-sz=.05 --num-iterations=1 --topk=5 --testing

"""
