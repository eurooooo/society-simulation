from log_schemas import AgentLog
from vllm_wrapper import BatchedLLM, SamplingParams
from dotenv import load_dotenv

load_dotenv()
model = "meta-llama/Llama-3.1-8B-Instruct"
# model = "meta-llama/Llama-3.2-3B-Instruct"
scenario = "different demographics political debate"
llm = BatchedLLM(model=model, max_model_len=8000, enable_prefix_caching=True)
sampling_params = SamplingParams(temperature=0.5, top_p=0.9, max_tokens=256, )

response = llm.generate(
    ["<|begin_of_text|><|start_header_id|>system<|end_header_id|>\nGiven an initial question and response from two political agents, score how much the agents agree with one another with values of -1, 0 or 1, where 1 means they agree and -1 means they disagree. Respond with an integer number only. Your response should contain no words, only a number, please.\n<|eot_id|><|start_header_id|>user<|end_header_id|>\nAgent 1: \n\nI think we need to invest in our social safety net programs, such as universal healthcare and education, to create a more equitable society, and I believe that's a worthwhile trade-off for the benefits they provide.\nAgent 2: \n\nI think taxes are too high and are stifling economic growth, and we need to reduce them to encourage entrepreneurship and job creation, rather than relying on the government to provide everything.\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"],
    # llama3.18B answer: -1
    # llama 3.2-3B answer: 0
    # ["<|begin_of_text|><|start_header_id|>system<|end_header_id|>\nGiven an initial question and response from two political agents, score how much the agents agree with one another with values of from -1 to 1, where 1 means they agree and -1 means they disagree. Respond with an integer number only. Your response should contain no words, only a number, please.\n<|eot_id|><|start_header_id|>user<|end_header_id|>\nAgent 1: \n\nI think taxes are too high. \nAgent 2: \n\nI disagree, I think we should raise taxes. \n"], # -1
    sampling_params)
print(response[0].outputs[0].text)