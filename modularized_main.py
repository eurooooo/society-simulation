import os
import datetime
import torch
import random
import argparse
from assign_pairs import assign_pairs1
from conversation import generate_conversation
from position_updates import update_positions
from starter_prompts import starter_prompts
from generate_vizualizaitons import generate_visualization_for_subdir
from dotenv import load_dotenv
import numpy as np


from vllm import SamplingParams
from vllm_wrapper import BatchedLLM


def parse_args():
    parser = argparse.ArgumentParser(description="Simulation of multi-agent interactions")
    parser.add_argument('--num-agents', type=int, required=True, help="Number of agents in the simulation")
    parser.add_argument('--step-sz', type=float, required=True, help="Step size for agent movement")
    parser.add_argument('--num-iterations', type=int, required=True, help="Number of iterations for the simulation")
    parser.add_argument('--topk', type=int, required=True, help="Top-k nearest neighbors for pairing")
    parser.add_argument('--simulation-timestamp', default=str(datetime.datetime.now()), help="Simulation timestamp")
    parser.add_argument('--gif', action='store_true', help="Generate a gif after running the simulation.")
    return parser.parse_args()


def main():
    llm = BatchedLLM(model="meta-llama/Llama-3.2-3B-Instruct", max_model_len=8000, enable_prefix_caching=True)
    sampling_params = SamplingParams(temperature=0.5, top_p=0.9, max_tokens=256, )

    # Group agent_bool, agent_loc to individual agent properties
    # Parse CLI args
    args = parse_args()
    top_level_dir = os.path.join("simulations", args.simulation_timestamp)
    os.makedirs(top_level_dir, exist_ok=True)

    num_agents = args.num_agents
    step_sz = args.step_sz
    num_iterations = args.num_iterations
    topk = args.topk

    # Initialize agents
    agents_bool = torch.rand(num_agents) > .5
    agents_loc = torch.rand(num_agents, 2)
    torch.save(agents_bool, os.path.join(top_level_dir, "bool.pt"))

    all_locations = []
    all_agreements = []
    agreement_metric = []
    metrics = [agreement_metric]

    for iter_idx in range(num_iterations):
        cur_agreements = []
        if iter_idx % 10 == 0:
            print(f"Iteration: {iter_idx}")

        # Pair agents and run conversations
        all_pairs = assign_pairs1(agents_loc, topk)

        # set up questions
        all_questions = [random.choice(starter_prompts) for _ in range(len(all_pairs))]

        all_replies, final_response = generate_conversation(llm, sampling_params, all_pairs, all_questions, agents_bool)
        # print("="*50+
        #       f"\nALL REPLIES"
        #       f"{all_replies}"
        #       f"\n\nALL FINAL RESPONSES"
        #       f"{final_response}")
        # update agent positions and appends agreement score using LLM judge
        agents_loc = update_positions(llm, sampling_params, final_response, all_questions, all_replies, all_pairs, agents_bool, agents_loc, step_sz, cur_agreements)
        all_locations.append(agents_loc.detach().clone().unsqueeze(0))

        all_agreements.append(torch.tensor(cur_agreements))
        agreement_metric.append(np.average(cur_agreements))
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

