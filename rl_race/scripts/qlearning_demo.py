import numpy as np
import random

# Environment setup
class GridWorld:
    def __init__(self, size):
        self.size = size
        self.state_space = size * size
        self.action_space = 4  # Up, Down, Left, Right
        self.goal = (size - 1, size - 1)
        self.reset()

    def reset(self):
        self.agent_pos = (0, 0)
        return self.agent_pos

    def step(self, action):
        x, y = self.agent_pos
        if action == 0:  # Up
            x = max(0, x - 1)
        elif action == 1:  # Down
            x = min(self.size - 1, x + 1)
        elif action == 2:  # Left
            y = max(0, y - 1)
        elif action == 3:  # Right
            y = min(self.size - 1, y + 1)

        self.agent_pos = (x, y)
        reward = 1 if self.agent_pos == self.goal else -0.01
        done = self.agent_pos == self.goal
        return self.agent_pos, reward, done

# Q-Learning Algorithm
class QLearningAgent:
    def __init__(self, env, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.env = env
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.q_table = np.zeros((env.size, env.size, env.action_space))

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, self.env.action_space - 1)  # Explore
        else:
            x, y = state
            return np.argmax(self.q_table[x, y])  # Exploit

    def update_q_table(self, state, action, reward, next_state):
        x, y = state
        next_x, next_y = next_state
        best_next_action = np.argmax(self.q_table[next_x, next_y])
        td_target = reward + self.gamma * self.q_table[next_x, next_y, best_next_action]
        td_error = td_target - self.q_table[x, y, action]
        self.q_table[x, y, action] += self.alpha * td_error

# Training the agent
def train_agent(env, agent, episodes):
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.update_q_table(state, action, reward, next_state)
            state = next_state
            total_reward += reward
        print(f"Episode {episode + 1}: Total Reward = {total_reward}")

# Main
if __name__ == "__main__":
    env = GridWorld(size=5)
    agent = QLearningAgent(env)
    train_agent(env, agent, episodes=1000)
