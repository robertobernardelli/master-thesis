import networkx as nx
import random

class Agent:
    def __init__(self, id):
        self.id = id
        self.household_connections = []
        self.workplace_connections = []
        self.friend_connections = []
        self.infected = False
    
    def decide(self):
        # TODO: Implement LLM-powered decision making
        self.go_to_work = random.random() < 0.5
        self.wear_mask = random.random() < 0.5
        self.social_activity = random.random() < 0.5
        self.take_private_transport = random.random() < 0.5
        
    
    def __repr__(self) -> str:
        return f"Agent #{self.id}"
    
    def __str__(self) -> str:
        return f"Agent #{self.id}"

class Network:
    """
    3-layer social network:
        1. Household network
        2. Workplace network
        3. Friendship network
    Each network is generated independently with BA algorithm
    Each node of the network is an Agent. So each agent will have in its memory the connections belonging to each of the three networks
    """
    def __init__(self, n_agents):
        # Generate the 3 layers of the network
        self.household_network = nx.barabasi_albert_graph(n_agents, 2)
        self.workplace_network = nx.barabasi_albert_graph(n_agents, 2)
        self.friendship_network = nx.barabasi_albert_graph(n_agents, 2)
        self.days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        self.current_day_of_week = self.days_of_week[0]
        self.simulation_days = 0

        # Initialise the agents
        self.agents = []
        for i in range(n_agents):
            self.agents.append(Agent(i))
        
        # Connect the agents
        for agent in self.agents:
            for connected_agent_index in self.household_network.neighbors(agent.id):
                agent.household_connections.append(self.agents[connected_agent_index])
            for connected_agent_index in self.workplace_network.neighbors(agent.id):
                agent.workplace_connections.append(self.agents[connected_agent_index])
            for connected_agent_index in self.friendship_network.neighbors(agent.id):
                agent.friend_connections.append(self.agents[connected_agent_index])
        
        self.infection_historical = []
        self.step = 0
    
    def social_contact(self, agent1, agent2):
        # TODO: Implement social contact model. For now there's a 50% chance of spreading the illness
        if agent1.infected and not agent2.infected:
            if random.random() < 0.5:
                agent2.infected = True
                print(f'Agent #{agent1} infected agent #{agent2}')
        elif agent2.infected and not agent1.infected:
            if random.random() < 0.5:
                agent1.infected = True
                print(f'Agent #{agent2} infected agent #{agent1}')
        
    
    def run_simulation(self, n_steps):
        for i in range(n_steps):

            # Agents decide their action
            for agent in self.agents:
                agent.decide()
            
            # Simulate contacts
            contacts = []
            for agent in self.agents:
                if self.current_day_of_week in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']:
                    if agent.go_to_work:
                        for workplace_agent in agent.workplace_connections:
                            if workplace_agent.go_to_work:
                                contacts.append((agent, workplace_agent))
                        if not agent.take_private_transport:
                            # random encounter with 5 other agents met on public transport
                            agents_going_to_work_public_transport = []
                            for a in self.agents:
                                if a.go_to_work and not a.take_private_transport:
                                    agents_going_to_work_public_transport.append(a)
                            agents_going_to_work_public_transport.remove(agent)
                            random_encounters = random.sample(agents_going_to_work_public_transport, 5)
                            for a in random_encounters:
                                contacts.append((agent, a))
                    
                    if agent.social_activity:
                        for friend in agent.friend_connections:
                            if friend.social_activity:
                                contacts.append((agent, friend))
            
            for contact in contacts:
                self.social_contact(contact[0], contact[1])
            
            # Sum total number of infected agents
            infected_agents = 0
            for agent in self.agents:
                if agent.infected:
                    infected_agents += 1
            self.infection_historical.append(infected_agents)
            print(f'Step {self.step}; infected: {infected_agents}')
            self.step += 1
            
