import numpy as np
import random
from env3rundivproduct import MultiAgentInvManagementDiv

class MeanFieldAgent:
    def __init__(self, agent_id, num_products):
        print("agent_id:", agent_id)
        self.agent_id = agent_id
        self.num_products = num_products
        self.q_table = np.zeros((100, 100))  # Example dimensions for Q-table
        self.epsilon = 0.1
        self.alpha = 0.1
        self.gamma = 0.9

    def choose_action(self, state, mean_field):
        if random.uniform(0, 1) < self.epsilon:
            return np.random.randint(0, 100, size=self.num_products)
        else:
            return np.argmax(self.q_table[state, mean_field])

    def update_q_table(self, state, mean_field, action, reward, next_state, next_mean_field):
        best_next_action = np.argmax(self.q_table[next_state, next_mean_field])
        td_target = reward + self.gamma * self.q_table[next_state, next_mean_field, best_next_action]
        td_delta = td_target - self.q_table[state, mean_field, action]
        self.q_table[state, mean_field, action] += self.alpha * td_delta

    def get_mean_field(self, actions):
        return np.mean(actions, axis=0)

def train_agents(env, num_episodes):
    node_names = env.node_names
    print("node names:", node_names)
    agents = [MeanFieldAgent(agent_id, env.num_products) for agent_id in node_names]

    for episode in range(num_episodes):
        all_state = env.reset()
        print("all state:", all_state)
        mean_field = np.zeros(env.num_products)
        
        for t in range(50):  # Assuming 50 timesteps per episode
            actions = {}
            states = []
            for agent in agents:
                print(agent.agent_id)
                state = all_state[agent.agent_id]
                print("state:", state)
                action = agent.choose_action(state, mean_field)
                actions[agent.agent_id] = action

            rewards = env.step(actions)
            
            next_mean_field = np.mean(actions, axis=0)
            for agent_id, agent in enumerate(agents):
                next_state = env.get_state(agent.agent_id)
                agent.update_q_table(states[agent_id], mean_field, actions[agent_id], rewards[agent_id], next_state, next_mean_field)
            
            mean_field = next_mean_field
        print("Episode: ", episode, "Reward: ", rewards)


config = {}
env = MultiAgentInvManagementDiv(config)
train_agents(env, 10)