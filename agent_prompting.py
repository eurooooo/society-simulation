from log_schemas import StaticAgentProperty2

gen_sys = (
        "You are going to engage in conversation with other politically minded individuals "
        "from the United States. Give your opinion on culture, economy, foreign affairs, "
        "and other political topics. Limit your response to 1-2 sentences."
    )

# def get_sys_prompt(is_gop):
# 
#     r_sys = "You are a conservative Republican in the United States.\nYou want limited government and a strong military.\n" + gen_sys
#     d_sys = "You are a liberal Democrat in the United States.\nYou want universal healthcare and gun regulations.\n" + gen_sys
# 
#     return r_sys if is_gop else d_sys

def get_sys_prompt(agent_property: StaticAgentProperty2):
    return agent_property.get_sys_prompt() + gen_sys
