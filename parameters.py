example_json = '''
            {
                "reasoning: : "Since the risk of infection is low, I will go to work by public transport without wearing a mask. I will also participate in a social activity with my friends."
                "go_to_work": true,
                "transport_public": true,
                "wear_mask": false,
                "social_activity": true
            }
            '''

# Proportions of agents with different behaviours (to be tuned)
CARELESS_PROPORTION = 0.1
SCARED_PROPORTION = 0.2
CAUTIOUS_PROPORTION = 0.7

# SEAIR parameters (days in each state)
LENGTH_EXPOSED = 2
LENGTH_ASYMPTOMATIC = 2
LENGTH_INFECTED = 5

# Define a dictionary mapping SEIR states to the corresponding attributes and next states
STATE_DICT = {
    'E': ('exposed_days', LENGTH_EXPOSED, 'A'),
    'A': ('asymptomatic_days', LENGTH_ASYMPTOMATIC, 'I'),
    'I': ('infected_days', LENGTH_INFECTED, 'R')
}

# Probability of infection when in contact with an infected agent
PROBABILITY_INFECTION = 0.1

# Mask efficiency # reduce the probability of infection by EFFICIENCY_MASK for each mask
# ex: if 2 agents are wearing masks, the probability of infection is reduced by EFFICIENCY_MASK^2
EFFICIENCY_MASK = 0.5

# Network building parameters (the M parameter is the number of edges to attach from a new node to existing nodes)
BARBARASI_M_WORKPLACE = 1 
BARBARASI_M_FRIENDSHIP = 1 

CONCURRENT_API_CALLS = 2