# example_json = '''
#             {
#                 "reasoning: : "Since the risk of infection is low, I will go to work by public transport without wearing a mask. I will also participate in a social activity with my friends."
#                 "go_to_work": true,
#                 "transport_public": true,
#                 "wear_mask": false,
#                 "social_activity": true
#             }
#             '''

# Proportions of agents with different behaviours (to be tuned)
CARELESS_PROPORTION = 0.25
SCARED_PROPORTION = 0.1
CAUTIOUS_PROPORTION = 0.65

# Probability of infection when in contact with an infected agent
PROBABILITY_INFECTION = 0.30

# Mask efficiency # reduce the probability of infection by EFFICIENCY_MASK for each mask
# ex: if 2 agents are wearing masks, the probability of infection is reduced by EFFICIENCY_MASK^2
EFFICIENCY_MASK = 0.47

# Network building parameters (the M parameter is the number of edges to attach from a new node to existing nodes)
# these paarameters match the questionnaire data
BARBARASI_M_WORKPLACE = 1 
BARBARASI_M_FRIENDSHIP = 1 

CONCURRENT_API_CALLS = 2

EXPOSED_TO_INFECTED_TRANSITION_PROB = 1/5.1 # day^-1
INFECTED_TO_RECOVERED_TRANSITION_PROB = 1/7 # day^-1
SYMPTOMATIC_PROB = 0.5 # probability of developing symptoms when infected

behaviour_dict = {
                'scared' : f'AGENT_NAME is very scared of the virus, and will only get out of home if there is zero or minimal risk of getting infected.',
                'cautious': f"AGENT_NAME will assess the situation, and will make his decisions depending on the risk. If there aren't many cases, he will not be worried, and act normally (e.g. go to work, go to the supermarket, socialize with friends).",
                'careless': f'AGENT_NAME is not cautious. AGENT_NAME is not afraid of the virus spreading in the city. AGENT_NAME doesn’t care about spreading the virus to his contacts, and disregards the well-being of others.'
            }

prompt_situation_assessment_template = """
CONTEXT: 
AGENT_NAME is AGENT_AGE years old. AGENT_NAME lives in Tamamushi City.
AGENT_NAME is currently aware that virus X spreading across the country. An active case can be infectious without feeling any symptoms (asymptomatic) and unknowingly spread the disease. From the newspaper, AGENT_NAME learns that there are currently NUMBER_SYMPTOMATIC_AGENTS confirmed infections cases (out of TOT_POPULATION population in Tamamushi).
AGENT_SHOWING_SYMPTOMS
FRIENDS_SYMPTOMATIC
COWORKERS_SYMPTOMATIC
HOUSEHOLD_SYMPTOMATIC

AGENT_NAME's perception of the virus: 
AGENT_BEHAVIOUR

Given this context, you must imagine what AGENT_NAME is thinking step-by-step as he would do.
"""


prompt_decision_making_template = f"""
AGENT_NAME's perception of the virus: 
AGENT_BEHAVIOUR

Reasoning of AGENT_NAME: 
REASONING_OUTPUT

Now, AGENT_NAME needs to make the following decisions:
- Go to work OR stay at home. His work cannot be done remotely.
- If AGENT_NAME goes to work, does he take the public transport (cheap but could expose you to infected people) or private transport (expensive, but safer)?
- After work, AGENT_NAME can decide if he wants to participate in a social activity with his friends.
- Does AGENT_NAME wear a mask today or not?

Given this context, AGENT_NAME weighs the risks and benefits of his decisions, and decides what to do. Imagine what AGENT_NAME would decide based on his reasoning.
"""


prompt_decision_extraction_template = """
I will give you a reasoning string and your task is to extract the boolean decisions out of a reasoning.

Example 1:
Reasoning: "Only 1% of the population is symptomatic, and all of my friends and coworkers are feeling good. Therefore, I believe the risk of getting the virus is low, and have decided to go to work by public transport. Wearing a mask will not be necessary. I will also participate in social activity with my friends."
Output:
{
"go_to_work": true,
"transport_public": true,
"wear_mask": false,
"social_activity": true
}

Example 2:
Reasoning: "The % of infectious cases is starting to become dangeourlsy high, but I still need to go to work. I will take a taxi to work to minimize the risk of exposure. I will wear a mask to protect myself. I will not participate in social activity with my friends as a precaution."
Output:
{
"go_to_work": true,
"transport_public": false,
"wear_mask": true,
"social_activity": false
}

Your turn now:
Reasoning: "DECISION_MAKING_OUTPUT"
"""


REFRESH_LLM_DECISION_EVERY_N_DAYS = 7

GPT_MODEL = "gpt-3.5-turbo-0125"