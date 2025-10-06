import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math

# Define the Grid Environment
class GridEnv:
    def __init__(self):
        self.grid_size = (10, 5)  # Size of the grid (10x5)
        self.state = (0, 0)  # Starting position of the agent
        self.goal = (9, 4)   # Goal position
        self.done = False    # Flag to check if the episode is done
        self.obstacles = [(4, 2), (5, 2), (6, 2), (2,6) , (4,8)]  # Obstacles in the grid
        self.prev_distance = np.linalg.norm(np.array(self.state) - np.array(self.goal))  # Initialize previous distance

    def reset(self):
        """Reset the environment to the initial state."""
        self.state = (0, 0)
        self.done = False
        self.prev_distance = np.linalg.norm(np.array(self.state) - np.array(self.goal))  # Reset previous distance
        return self._get_state_index()

    def step(self, action):
        """Take an action and return the new state, reward, done flag, and additional info."""
        if self.done:
            return self._get_state_index(), 0, self.done, {}

        x, y = self.state
        if action == 0 and x > 0:     # Up
            x -= 1
        elif action == 1 and y < self.grid_size[1] - 1:  # Right
            y += 1
        elif action == 2 and x < self.grid_size[0] - 1:  # Down
            x += 1
        elif action == 3 and y > 0:     # Left
            y -= 1

        new_state = (x, y)
        if new_state in self.obstacles:  # Handle obstacle collision
            reward = -10
            self.prev_distance = np.linalg.norm(np.array(new_state) - np.array(self.goal))  # Update previous distance
        else:
            current_distance = np.linalg.norm(np.array(new_state) - np.array(self.goal))
            if current_distance < self.prev_distance:  # Bot is closer to the goal
                reward = 2
            elif current_distance > self.prev_distance:  # Bot is farther from the goal
                reward = -2
            else:
                reward = -0.1  # Small penalty for moving without progress

            self.state = new_state
            self.prev_distance = current_distance  # Update previous distance

        self.done = self.state == self.goal
        return self._get_state_index(), reward, self.done, {}

    def _get_state_index(self):
        """Convert the (x, y) state into a single integer index."""
        x, y = self.state
        return x * self.grid_size[1] + y

# Define the PPO Agent
class PPOAgent:
    def __init__(self, state_dim, action_dim, clip_param=0.4):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.clip_param = clip_param
        self.policy = np.ones((state_dim, action_dim)) / action_dim  # Initialize policy to uniform probabilities
        self.value_function = np.zeros(state_dim)  # Initialize value function to zeros
        self.lr = 0.01  # Learning rate
        self.epsilon = 1e-8  # Small value to avoid division by zero

    def act(self, state):
        """Choose an action based on the current policy."""
        prob = self.policy[state]
        prob = np.clip(prob, self.epsilon, 1.0)  # Ensure probabilities are within a valid range
        prob /= prob.sum()  # Ensure probabilities sum to 1
        return np.random.choice(self.action_dim, p=prob)

    def evaluate(self, state, action):
        """Evaluate the log probability of an action and the value function."""
        prob = self.policy[state]
        action_prob = prob[action]
        value = self.value_function[state]
        return np.log(action_prob), value

    def ppo_update(self, trajectories):
        """Update the policy and value function using PPO."""
        states, actions, rewards, dones = zip(*trajectories)
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        dones = np.array(dones)

        # Calculate the returns (cumulative rewards)
        returns = np.zeros_like(rewards)
        G = 0
        for t in reversed(range(len(rewards))):
            G = rewards[t] + (0 if dones[t] else 0.99 * G)  # Assuming discount factor of 0.99
            returns[t] = G

        advantages = returns - self.value_function[states]  # Calculate advantages
        old_action_probs = np.array([self.policy[s][a] for s, a in zip(states, actions)])

        for _ in range(10):  # Number of PPO epochs
            new_action_probs = np.array([self.policy[s][a] for s, a in zip(states, actions)])
            ratios = new_action_probs / (old_action_probs + self.epsilon)  # Add epsilon to avoid division by zero
            clipped_ratios = np.clip(ratios, 1 - self.clip_param, 1 + self.clip_param)
            policy_loss = -np.mean(np.minimum(ratios * advantages, clipped_ratios * advantages))

            # Update policy and value function
            for s, a, advantage, return_ in zip(states, actions, advantages, returns):
                self.policy[s][a] += self.lr * policy_loss
                self.value_function[s] += self.lr * (return_ - self.value_function[s])

        # Normalize policy probabilities to ensure they sum to 1
        for s in np.unique(states):
            self.policy[s] = np.clip(self.policy[s], self.epsilon, 1.0)  # Clamp values to avoid NaN
            self.policy[s] /= self.policy[s].sum()  # Normalize probabilities


# Training loop for the PPO Agent
def train_ppo(agent, env, episodes):
    step_limit = 10  # Maximum steps per episode to avoid infinite loops

    for episode in range(episodes):
        state = env.reset()
        trajectories = []
        total_reward = 0

        for step in range(step_limit):
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            trajectories.append((state, action, reward, done))
            state = next_state
            total_reward += reward

            # Print details for each step
            print(f"Episode {episode}, Step {step}: State={state}, Action={action}, Reward={reward}")

            if done:
                break

        agent.ppo_update(trajectories)

        if episode % 10 == 0:
            print(f"Episode {episode}: Total Reward: {total_reward}")

# Function to visualize the learned policy
def visualize_policy(agent, env):
    fig, ax = plt.subplots()
    grid_size = env.grid_size
    ax.set_xlim(-0.5, grid_size[1] - 0.5)
    ax.set_ylim(-0.5, grid_size[0] - 0.5)

    # Add arrows to indicate the policy
    for x in range(grid_size[0]):
        for y in range(grid_size[1]):
            state = x * grid_size[1] + y
            action = np.argmax(agent.policy[state])

            if action == 0:  # Up
                dx, dy = 0, 0.3
            elif action == 1:  # Right
                dx, dy = 0.3, 0
            elif action == 2:  # Down
                dx, dy = 0, -0.3
            elif action == 3:  # Left
                dx, dy = -0.3, 0

            ax.arrow(y, x, dx, dy, head_width=0.1, head_length=0.1, fc='black', ec='black')

    # Plot the grid
    for i in range(grid_size[0] + 1):
        ax.plot([-0.5, grid_size[1] - 0.5], [i - 0.5, i - 0.5], color='black')
    for i in range(grid_size[1] + 1):
        ax.plot([i - 0.5, i - 0.5], [-0.5, grid_size[0] - 0.5], color='black')

    # Add obstacles
    for obs in env.obstacles:
        rect = patches.Rectangle((obs[1] - 0.5, obs[0] - 0.5), 1, 1, linewidth=1, edgecolor='r', facecolor='gray')
        ax.add_patch(rect)

    # Mark start and goal
    ax.text(0, 0, 'Start', ha='center', va='center', color='blue', fontweight='bold')
    ax.text(env.goal[1], env.goal[0], 'Goal', ha='center', va='center', color='green', fontweight='bold')

    ax.set_title('Learned Policy')
    plt.gca().invert_yaxis()
    plt.show()

# Main execution
if __name__ == "__main__":
    grid_size = (10, 5)
    env = GridEnv()
    agent = PPOAgent(state_dim=grid_size[0] * grid_size[1], action_dim=4)

    # Train the agent
    train_ppo(agent, env, episodes=50)  # Use fewer episodes to test quickly

    # Visualize the learned policy in the main thread
    visualize_policy(agent, env)
