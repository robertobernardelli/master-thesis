import networkx as nx
import random
import seaborn as sns
import matplotlib.pyplot as plt

class Agent:
    def __init__(self, id):
        self.id = id
        self.household_connections = []
        self.workplace_connections = []
        self.friend_connections = []
        
        # SEAIR State:
        self.seir_state = 'S'

        # Keep track of days in each state, for transition
        self.exposed_days = 0
        self.asymptomatic_days = 0
        self.infected_days = 0
    
    def decide(self):
        #### TODO: Implement LLM-powered decision making. For now the agent isolates if infected
        high_probability = 1
        low_probability = 0
        if self.seir_state == 'I':
            self.go_to_work = random.random() < low_probability
            self.social_activity = random.random() < low_probability
            self.wear_mask = random.random() < high_probability
            self.take_private_transport = random.random() < high_probability
        else:
            self.go_to_work = random.random() < 0.9 
            self.social_activity = random.random() < 0.3 
            self.wear_mask = random.random() < low_probability
            self.take_private_transport = random.random() < 0.2
        
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

class Network:

    def __init__(self, n_agents):
        # Generate the 3 layers of the network
        self.household_network = generate_household_network(n_agents)
        self.workplace_network = nx.barabasi_albert_graph(n_agents, 1)
        self.friendship_network = nx.barabasi_albert_graph(n_agents, 1)

        # Days of the week are needed to simulate the work days
        self.days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        self.current_day_of_week = self.days_of_week[0]
        self.simulation_days = 0

        # Initialise the agents
        self.agents = []
        for i in range(n_agents):
            self.agents.append(Agent(i))
        
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

        
    def plot_trends(self):
        """
        Generate a plot of the number of agents in each SEAIR state over time
        """
        sns.lineplot(x=self.days_infection_historical, y=self.infected_historical, label='Infected')
        sns.lineplot(x=self.days_infection_historical, y=self.susceptible_historical, label='Susceptible')
        sns.lineplot(x=self.days_infection_historical, y=self.exposed_historical, label='Exposed')
        sns.lineplot(x=self.days_infection_historical, y=self.recovered_historical, label='Recovered')
        sns.lineplot(x=self.days_infection_historical, y=self.asymptomatic_historical, label='Asymptomatic')
        plt.show()
        
    
    def social_contact(self, agent1, agent2, contact_type):
        # TODO: Implement infection spreading model. For now there's a 50% chance of spreading the illness
        
        # Check patient zero contacts:
        #  if agent1.id == 90 or agent2.id == 90:
        #     print(f'Agent #{agent1} contacted agent #{agent2} ({contact_type})')
        
        n_wearing_masks = 0
        
        for agent in [agent1, agent2]:
            if agent.wear_mask:
                n_wearing_masks += 1

        if agent1.seir_state in ['I', 'A'] and agent2.seir_state == 'S':
            if n_wearing_masks == 2:
                if random.random() < 0.2:
                    agent2.seir_state = 'E'
                    print(f'Agent #{agent1} infected agent #{agent2} ({contact_type})')
            elif n_wearing_masks == 1:
                if random.random() < 0.2:
                    agent2.seir_state = 'E'
                    print(f'Agent #{agent1} infected agent #{agent2} ({contact_type})')
            else:
                if random.random() < 0.4:
                    agent2.seir_state = 'E'
                    print(f'Agent #{agent1} infected agent #{agent2} ({contact_type})')
        
        elif agent1.seir_state == 'S' and agent2.seir_state in ['I', 'A']:
            if n_wearing_masks == 2:
                if random.random() < 0.2:
                    agent1.seir_state = 'E'
                    print(f'Agent #{agent2} infected agent #{agent1} ({contact_type})')
            elif n_wearing_masks == 1:
                if random.random() < 0.2:
                    agent1.seir_state = 'E'
                    print(f'Agent #{agent2} infected agent #{agent1} ({contact_type})')
            else:
                if random.random() < 0.4:
                    agent1.seir_state = 'E'
                    print(f'Agent #{agent2} infected agent #{agent1} ({contact_type})')
    
    def run_simulation(self, n_steps):
        for i in range(n_steps):
            print(f'Step {i} of {n_steps}')

            # if there are no infected/asymtpomatic agents, we stop the simulation
            if len([agent for agent in self.agents if agent.seir_state in ['I', 'A']]) == 0:
                print('No more infected agents. Stopping simulation.')
                break

            # Update the SEIR state of the agents
            for agent in self.agents:
                if agent.seir_state == 'E':
                    agent.exposed_days += 1
                    if agent.exposed_days == 2:
                        agent.seir_state = 'A'
                elif agent.seir_state == 'A':
                    agent.asymptomatic_days += 1
                    if agent.asymptomatic_days == 2:
                        agent.seir_state = 'I'
                elif agent.seir_state == 'I':
                    agent.infected_days += 1
                    if agent.infected_days == 5:
                        agent.seir_state = 'R'

            # Agents decide their action with LLM
            for agent in self.agents:
                agent.decide()
            
            # Simulate contacts
            contacts = []
            for agent in self.agents:
                if agent.seir_state in ['I', 'A']:
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
            print(f'Removed {len(contacts) - len(new_contacts)} duplicate contacts')
            contacts = new_contacts

            for contact in contacts:
                self.social_contact(contact[0], contact[1], contact[2])
            
            # Telemetry
            self.days_infection_historical.append(self.simulation_days)

            # Total infected agents:
            
            
            self.susceptible_historical.append(len([agent for agent in self.agents if agent.seir_state == 'S']))
            self.exposed_historical.append(len([agent for agent in self.agents if agent.seir_state == 'E']))
            self.asymptomatic_historical.append(len([agent for agent in self.agents if agent.seir_state == 'A']))
            self.infected_historical.append(len([agent for agent in self.agents if agent.seir_state == 'I']))
            self.recovered_historical.append(len([agent for agent in self.agents if agent.seir_state == 'R']))
            
            self.step += 1

            # Update the day of the week
            self.simulation_days += 1
            self.current_day_of_week = self.days_of_week[self.simulation_days % 7]