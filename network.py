import networkx as nx
import random
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

api_key = open('key.txt').read().strip()
from openai import OpenAI
client = OpenAI(api_key=api_key)

import concurrent.futures

from parameters import *

import os 
import datetime
import pandas as pd

import json

class Agent:
    def __init__(self, id, name, age, behaviour, behaviour_model):
        self.id = id
        self.name = name
        self.age = age
        self.behaviour = behaviour # scared, cautious, careless
        self.behaviour_model = behaviour_model
        self.household_connections = []
        self.workplace_connections = []
        self.friend_connections = []
        
        # SEAIR State:
        self.seir_state = 'S'

        # Keep track of days in each state, for transition
        self.exposed_days = 0
        self.asymptomatic_days = 0
        self.infected_days = 0

        # Keep track of all the states and decisions made by the agent at each step
        self.telemetry = []
        self.prompt = ''

    def decide(self, perc_pop_infected):

        # Recovered agents do not take any actions, as they do not influence the spread of the virus anymore
        if self.seir_state == 'R':
            self.go_to_work = True
            self.social_activity = True
            self.wear_mask = False
            self.take_private_transport = False
            self.reasoning = 'Agent is Recovered and does not take any actions'
            return
            
        # get list of friends which are symptomatic
        friends_symptomatic = [friend.name for friend in self.friend_connections if friend.seir_state in ['I']]
        n_friends_all = len(self.friend_connections)
        if len(friends_symptomatic) == 0:
            friends_str = ''
        else:
            friends_str = f'{self.name} has {n_friends_all} friends. Of these friends, {", ".join(friends_symptomatic)} are currently infected and symptomatic'
        self.n_friends_all = n_friends_all
        self.n_friends_symptomatic = len(friends_symptomatic)

        coworkers_symptomatic = [coworker.name for coworker in self.workplace_connections if coworker.seir_state in ['I']]
        n_coworkers_all = len(self.workplace_connections)
        if len(coworkers_symptomatic) == 0:
            coworkers_str = ''
        else:
            coworkers_str = f'{self.name} works with {n_coworkers_all} people. Of these people, {", ".join(coworkers_symptomatic)} are currently infected and symptomatic'
        self.n_coworkers_all = n_coworkers_all
        self.n_coworkers_symptomatic = len(coworkers_symptomatic)


        household_symptomatic = [household.name for household in self.household_connections if household.seir_state in ['I']]
        n_household_all = len(self.household_connections)
        if len(household_symptomatic) == 0:
            household_str = ''
        else:
            household_str = f'{self.name} lives with {n_household_all} people. Of these people, {", ".join(household_symptomatic)} are currently infected and symptomatic'
        self.n_household_all = n_household_all
        self.n_household_symptomatic = len(household_symptomatic)

        if self.behaviour_model == 'ABM-generative':
            # Generate the agent's decision
            
            behaviour_dict = {
                'scared' : f'{self.name} is very scared of the virus, and will only get out of home if there is zero or minimal risk of getting infected. ',
                'cautious': f'{self.name} will assess the situation, and will make his decisions depending on the risk. For example, if the risk is low or not significant, he will go out without a mask (for example if the percentage of infected people is relatively low).',
                'careless': f'{self.name} is not afraid of the virus spreading in the city. {self.name} doesn’t care about spreading the virus to his contacts, and disregards the well-being of others.'
            }
            behaviour_prompt = behaviour_dict[self.behaviour]

            # if the agent is showing symptoms, put it in the prompt
            if self.seir_state in ['I']:
                showing_symptoms = f'{self.name} is currently showing symptoms of virus X, and could spread it to others.'
            else:
                showing_symptoms = ''

            prompt = f'''
                        CONTEXT:
                        {self.name} is {self.age} years old. {self.name} lives in Tamamushi City.
                        {showing_symptoms}
                        {self.name} is currently aware that virus X spreading across the country. From the newspaper, {self.name} learns that {int(perc_pop_infected*100)}% of the population in Tamamushi City has been infected with X.
                        {friends_str}
                        {coworkers_str}
                        {household_str}
                        {behaviour_prompt}

                        Given {self.name}'s belief about the virus X and context, he needs to make the following decisions:
                        - Go to work OR stay at home. His work cannot be done remotely.
                        - If {self.name} goes to work, does he take the public transport (cheap but could expose you to infected people) or taxi (expensive, but safer)?
                        - After work, {self.name} can decide if he wants to participate in a social activity with his friends.
                        - Does {self.name} wear a mask today or not?

                        Write an output json that briefly summarises the reasoning and states the decisions {self.name} made. 

                        Example of a generic output json:
                        {example_json} 
                    '''
            
            self.prompt = prompt # store for telemetry
            
            # read cache from cache.json and close the file
            with open('cache.json', 'r') as f:
                cache = json.load(f)

            if prompt in cache:
                # If it is, return the corresponding response
                response_str = cache[prompt]
                print('Response from cache!')
            else:
                # Get the response from the API
                response = client.chat.completions.create(
                        model="gpt-3.5-turbo-1106",
                        response_format={ "type": "json_object" },
                        #seed = 0,  # <<<--- uncomment this line to get deterministic results (Reproducibility)
                        messages=[
                            {"role": "system", "content": f"Impersonate {self.name}. Output his decisions in JSON format."},
                            {"role": "user", "content": prompt},
                        ]
                        )
                response_str = response.choices[0].message.content

                # re-open the file to read the previous cache and update it
                with open('cache.json', 'r') as f:
                    cache = json.load(f)

                # update the cache with the new response
                cache[prompt] = response_str

                # write the updated cache to the file
                with open('cache.json', 'w') as f:
                    json.dump(cache, f)

            # extract the decisions from the response
            try:
                decisions = response_str
                decisions = decisions.replace('true', 'True')
                decisions = decisions.replace('false', 'False')

                # in most of the cases where the model output none or null, he means False (self-isolate):
                decisions = decisions.replace('none', 'False') 
                decisions = decisions.replace('null', 'False')
                decisions = decisions.replace('None', 'False') 
                decisions = decisions.replace('Null', 'False')
                
                decisions = eval(decisions)

                try:
                    self.go_to_work = decisions['go_to_work']
                except: # default decision: go to work
                    self.go_to_work = True

                try:
                    self.social_activity = decisions['social_activity']
                except: # default decision: participate in social activity
                    self.social_activity = True

                try:
                    self.wear_mask = decisions['wear_mask']
                except: # default decision: don't wear mask
                    self.wear_mask = False

                try:
                    self.take_private_transport = decisions['transport_public']
                except: # default decision: take public transport
                    self.take_private_transport = False

                try:
                    self.reasoning = decisions['reasoning']
                except: # default reasoning
                    self.reasoning = 'NO REASONING PROVIDED'
            
            except Exception as e:
                # Error in parsing the response. Using default decisions
                self.go_to_work = True
                self.social_activity = True
                self.wear_mask = False
                self.take_private_transport = False
                self.reasoning = 'ERROR: LLM RESPONSE PARSING FAILED'

        elif self.behaviour_model == 'ABM-isolation':
            if self.seir_state == 'I':
                self.go_to_work = False
                self.social_activity = False
                self.wear_mask = True
                self.take_private_transport = True
                self.reasoning = 'ABM-isolation: Agent is infected, staying at home and isolating'
            else: 
                self.go_to_work = True
                self.social_activity = True
                self.wear_mask = False
                self.take_private_transport = False
                self.reasoning = 'ABM-isolation: Agent is not infected, going to work and socializing'
        
        elif self.behaviour_model == 'ABM-normal':
            self.go_to_work = True
            self.social_activity = True
            self.wear_mask = False
            self.take_private_transport = False
            self.reasoning = 'ABM-normal: Agent is going to work and socializing, as always'
        else:
            raise NotImplementedError(f'{self.behaviour_model} model not implemented')

        
    def __repr__(self) -> str:
        return f"Agent #{self.id}"
    
    def __str__(self) -> str:
        return f"Agent #{self.id}"

def generate_household_network(n_agents):
    """
    Generate a random household network, where each agent belongs to a 
    household of 1 to 5 agents. Each agent is connected to all other 
    agents in the same household.
    It can happen that the agent is not connected to any other agent in 
    the household, if the household size is 1.
    """
    household_network = nx.Graph()
    agents = list(range(n_agents))
    random.shuffle(agents)

    while agents:
        household_size = min(random.randint(1, 5), len(agents))
        household_agents = [agents.pop() for _ in range(household_size)]
        for agent in household_agents:
            for other_agent in household_agents:
                if agent != other_agent:
                    household_network.add_edge(agent, other_agent)
                else:
                    household_network.add_node(agent)

    return household_network

def random_age():
    # outputs a random age between 18 and 80 using a normal distribution with mean 40 and std 15, capped at 18 and 80
    return max(18, min(80, int(np.random.normal(40, 15))))

class Network:

    def __init__(self, n_agents, behaviour_model):
        
        if not behaviour_model in ['ABM-normal', 'ABM-isolation', 'ABM-generative']:
            raise NotImplementedError(f'{behaviour_model} model not implemented')

        # Generate the 3 layers of the network
        self.household_network = generate_household_network(n_agents)
        self.workplace_network = nx.barabasi_albert_graph(n_agents, BARBARASI_M_WORKPLACE)
        self.friendship_network = nx.barabasi_albert_graph(n_agents, BARBARASI_M_FRIENDSHIP)

        self.days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        self.current_day_of_week = self.days_of_week[0]
        self.simulation_days = 0
        self.perc_pop_infected = 0

        # Load first-names.txt
        with open('first-names.txt', 'r') as f:
            first_names = f.readlines()
        first_names = [name.strip() for name in first_names]

        # Initialise the agents
        self.agents = []
        for i in range(n_agents):
            name = random.choice(first_names)
            age = random_age()

            behaviour = np.random.choice(['scared', 'cautious', 'careless'], p=[SCARED_PROPORTION, CAUTIOUS_PROPORTION, CARELESS_PROPORTION])
            self.agents.append(Agent(i, name, age, behaviour, behaviour_model))
        
        # Connect the agents, using the 3 networks
        for agent in self.agents:
            for connected_agent_index in self.household_network.neighbors(agent.id):
                agent.household_connections.append(self.agents[connected_agent_index])
            for connected_agent_index in self.workplace_network.neighbors(agent.id):
                agent.workplace_connections.append(self.agents[connected_agent_index])
            for connected_agent_index in self.friendship_network.neighbors(agent.id):
                agent.friend_connections.append(self.agents[connected_agent_index])
    
        self.step = 0

        # self initialise list of days and number of infected agents, for telemetry and trend plotting
        self.days_infection_historical = []
        self.susceptible_historical = []
        self.exposed_historical = []
        self.infected_historical = []
        self.recovered_historical = []
        self.asymptomatic_historical = []
        self.mobility_historical = []
        self.avg_contacts_historical = []

        self.n_agents = n_agents
        self.behaviour_model = behaviour_model
    
    def output_telemetry(self):
        # create a folder named with main settings and current datetime string
        self.output_folder_name = f'output/{self.behaviour_model}_{self.n_agents}_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
        os.makedirs(self.output_folder_name)

        # save a csv with the network-level telemetry as plotted in plot_trends. create a pd dataframe and save it as csv
        df = pd.DataFrame({
            'day': self.days_infection_historical,
            'susceptible': self.susceptible_historical,
            'exposed': self.exposed_historical,
            'infected': self.infected_historical,
            'recovered': self.recovered_historical,
            'asymptomatic': self.asymptomatic_historical,
            'mobility': self.mobility_historical
        })

        df.to_csv(f'{self.output_folder_name}/network_telemetry.csv', index=False)

        # save a txt file with the telemetry of all the agents, in a single file
        with open(f'{self.output_folder_name}/agents_telemetry.txt', 'w') as f:
            for agent in self.agents:
                for telemetry in agent.telemetry:
                    f.write(f"{telemetry}\n")

        self.plot_trends()
        
    def plot_trends(self):
        # Plot the trends of the infection
        sns.lineplot(x=self.days_infection_historical, y=self.infected_historical, label='Infected')
        sns.lineplot(x=self.days_infection_historical, y=self.susceptible_historical, label='Susceptible')
        sns.lineplot(x=self.days_infection_historical, y=self.exposed_historical, label='Exposed')
        sns.lineplot(x=self.days_infection_historical, y=self.recovered_historical, label='Recovered')
        sns.lineplot(x=self.days_infection_historical, y=self.asymptomatic_historical, label='Asymptomatic')
        plt.xlabel('Day')
        plt.ylabel('Number of agents')
        plt.title('Infection trends')
        plt.savefig(f'{self.output_folder_name}/infection_trends.png')
        plt.show()
        

        # compute daily new cases, by computing difference between susceptible agents and the sum of the other states
        daily_new_cases = [0]
        for i in range(1, len(self.infected_historical)):
            daily_new_cases.append(self.susceptible_historical[i-1] - self.susceptible_historical[i])
        sns.lineplot(x=self.days_infection_historical, y=daily_new_cases, label='Daily new cases')
        plt.xlabel('Day')
        plt.ylabel('Number of new cases')
        plt.title('Daily new cases')
        plt.savefig(f'{self.output_folder_name}/daily_new_cases.png')
        plt.show()

        # Plot the mobility trend
        sns.lineplot(x=self.days_infection_historical, y=self.mobility_historical, label='Mobility')
        plt.xlabel('Day')
        plt.ylabel('Number of agents not self-isolating')
        plt.ylim(0, self.n_agents)  # Set the limits of the y-axis
        plt.title('Mobility trend')
        plt.savefig(f'{self.output_folder_name}/mobility_trend.png')
        plt.show()

        # Plot the average number of contacts per agent
        sns.lineplot(x=self.days_infection_historical, y=self.avg_contacts_historical, label='Average number of contacts per agent')
        plt.xlabel('Day')
        plt.ylabel('Average number of contacts per agent')
        plt.title('Average number of contacts per agent')
        plt.savefig(f'{self.output_folder_name}/avg_contacts_trend.png')
        plt.show()
    
    def social_contact(self, agent1, agent2, contact_type):
        n_wearing_masks = sum(agent.wear_mask for agent in [agent1, agent2]) # can be 0, 1 or 2
        p = PROBABILITY_INFECTION * (EFFICIENCY_MASK ** n_wearing_masks)

        for susceptible, infectious in [(agent1, agent2), (agent2, agent1)]:

            # Check if either agent is susceptible and the other is infectious
            if susceptible.seir_state == 'S' and infectious.seir_state in ['I', 'A']:

                if random.random() < p:
                    susceptible.seir_state = 'E'
                    print(f'Agent #{infectious} infected agent #{susceptible} ({contact_type})')
                    break  # Exit loop once infection occurs
    
    def run_simulation(self, n_steps):
        for i in range(n_steps):
            print(f'Step {i} of {n_steps} ({self.current_day_of_week})')

            # if there are no infected/asymtpomatic agents, we stop the simulation
            if len([agent for agent in self.agents if agent.seir_state in ['I', 'A']]) == 0:
                print('No more infected agents. Stopping simulation.')
                break
                
            # Telemetry
            self.days_infection_historical.append(self.simulation_days)
            self.susceptible_historical.append(len([agent for agent in self.agents if agent.seir_state == 'S']))
            self.exposed_historical.append(len([agent for agent in self.agents if agent.seir_state == 'E']))
            self.asymptomatic_historical.append(len([agent for agent in self.agents if agent.seir_state == 'A']))
            self.infected_historical.append(len([agent for agent in self.agents if agent.seir_state == 'I']))
            self.recovered_historical.append(len([agent for agent in self.agents if agent.seir_state == 'R']))


            # Update the SEIR state of the agents
            for agent in self.agents:
                if agent.seir_state in STATE_DICT:
                    attr, length, next_state = STATE_DICT[agent.seir_state]
                    setattr(agent, attr, getattr(agent, attr) + 1)
                    if getattr(agent, attr) > length:
                        agent.seir_state = next_state

            # Update the percentage of infected agents (will be used by the agents to decide their actions)
            self.perc_pop_infected = len([agent for agent in self.agents if agent.seir_state in ['I']]) / len(self.agents)

            # Remove previous day's decisions for all agents and default to no decision (needed in case API fails)
            for agent in self.agents:
                agent.go_to_work = True
                agent.social_activity = True
                agent.wear_mask = False
                agent.take_private_transport = False
                agent.reasoning = 'NO DECISION YET'
                agent.prompt = 'NO PROMPT YET'

            # # Parallelized agent decision-making
            # with concurrent.futures.ThreadPoolExecutor(max_workers=CONCURRENT_API_CALLS) as executor:
            #     # Submit agent decisions in parallel
            #     decision_futures = [executor.submit(agent.decide, self.perc_pop_infected) for agent in self.agents]
                
            #     # Wait for all decisions to complete
            #     concurrent.futures.wait(decision_futures)

            for agent in self.agents:
                agent.decide(self.perc_pop_infected)
            
            # Compute the percentage of agents that failed to make a decision (reasoning = 'NO DECISION YET')
            perc_failed_decision = len([agent for agent in self.agents if agent.reasoning == 'NO DECISION YET']) / len(self.agents)
            if perc_failed_decision > 0.2:
                raise Exception(f'Rate Limit Exceeded: {round(perc_failed_decision*100, 2)}% of API calls failed. Stopping simulation.')

            # Add telemetry to the agents
            for agent in self.agents:
                new_telemetry = {
                    'name': agent.name,
                    'seir_state': agent.seir_state,
                    'behaviour': agent.behaviour,
                    'reasoning': agent.reasoning,
                    'age': agent.age,
                    'id': agent.id,
                    'day': self.simulation_days,
                    'perc_pop_infected': self.perc_pop_infected,
                    'go_to_work': agent.go_to_work,
                    'take_private_transport': agent.take_private_transport,
                    'social_activity': agent.social_activity,
                    'wear_mask': agent.wear_mask,
                    'n_friends_all': agent.n_friends_all,
                    'n_friends_symptomatic': agent.n_friends_symptomatic,
                    'n_coworkers_all': agent.n_coworkers_all,
                    'n_coworkers_symptomatic': agent.n_coworkers_symptomatic,
                    'n_household_all': agent.n_household_all,
                    'n_household_symptomatic': agent.n_household_symptomatic,
                    'prompt': agent.prompt
                }
                print(new_telemetry)
                agent.telemetry.append(new_telemetry)
            
            # Simulate contacts
            contacts = []
            for agent in self.agents:
                #if agent.seir_state in ['I', 'A']:
                if self.current_day_of_week in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']:
                    if agent.go_to_work:
                        n_encounters_work = 2
                        try:
                            random_encounters = random.sample(agent.workplace_connections, n_encounters_work)
                        except ValueError:
                            random_encounters = []
                        for a in random_encounters:
                            contacts.append((agent, a, 'work'))

                        if not agent.take_private_transport: # if the agent takes public transport
                            
                            agents_going_to_work_public_transport = []
                            for a in self.agents:
                                if a.go_to_work and not a.take_private_transport:
                                    agents_going_to_work_public_transport.append(a)
                            agents_going_to_work_public_transport.remove(agent)

                            n_encounters_public_transport = 1
                            try:
                                random_encounters = random.sample(agents_going_to_work_public_transport, 
                                                                n_encounters_public_transport)
                            except ValueError:
                                random_encounters = agents_going_to_work_public_transport
                            for a in random_encounters:
                                contacts.append((agent, a, 'public_transport'))
                    
                if agent.social_activity: # if the agent decided to have a social activity today
                    for friend in agent.friend_connections:
                        if friend.social_activity:  # register friends who also decided to have a social activity
                            contacts.append((agent, friend, 'social'))
                    
                    # Household contacts
                    n_encounters_household = 2
                    try:
                        random_encounters = random.sample(agent.household_connections, n_encounters_household)
                    except ValueError:
                        random_encounters = []
                    for a in random_encounters:
                        contacts.append((agent, a, 'household'))
            
            # remove duplicates in contacts (only one contact per pair of agents, regardless of order of the pair)
            new_contacts = []
            for contact in contacts:
                agent1, agent2, contact_type = contact
                if (agent2, agent1, contact_type) not in new_contacts and (agent1, agent2, contact_type) not in new_contacts:
                    new_contacts.append(contact)
            contacts = new_contacts

            for contact in contacts:
                self.social_contact(contact[0], contact[1], contact[2])

            # compute average number of contacts per agent
            n_contacts = len(contacts)
            n_agents = len(self.agents)
            avg_contacts = n_contacts/n_agents
            self.avg_contacts_historical.append(avg_contacts)
            
            self.step += 1

            # Compute Mobility: number of agents that decided to go to work and/OR have a social activity
            self.mobility_historical.append(len([agent for agent in self.agents if agent.go_to_work or agent.social_activity]))

            # Update the day of the week
            self.simulation_days += 1
            self.current_day_of_week = self.days_of_week[self.simulation_days % 7]
        
        self.output_telemetry()