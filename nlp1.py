import gym

# Create the CartPole environment
env = gym.make("CartPole-v1")

num_episodes = 10

for episode in range(num_episodes):
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        env.render()  # Comment this if you don't want to render
        action = env.action_space.sample()  # Random action
        next_state, reward, done, info = env.step(action)
        total_reward += reward

    print(f"Episode {episode + 1}: Total Reward = {total_reward}")

env.close()


import numpy as np

# Define a simple MDP
states = [0, 1, 2]
actions = [0, 1]  # Let's say: 0 = left, 1 = right
P = {
    0: {
        0: [(1.0, 0, 0)],      # From state 0, taking action 0 -> state 0, reward 0
        1: [(1.0, 1, 5)],      # From state 0, action 1 -> state 1, reward 5
    },
    1: {
        0: [(1.0, 0, 0)],
        1: [(1.0, 2, 10)],
    },
    2: {
        0: [(1.0, 2, 0)],
        1: [(1.0, 2, 0)],
    }
}
gamma = 0.9  # Discount factor
theta = 1e-4  # Threshold

# Initialize value function
V = np.zeros(len(states))

def value_iteration(P, states, actions, gamma, theta):
    while True:
        delta = 0
        for s in states:
            v = V[s]
            action_values = []
            for a in actions:
                action_value = 0
                for prob, next_state, reward in P[s][a]:
                    action_value += prob * (reward + gamma * V[next_state])
                action_values.append(action_value)
            V[s] = max(action_values)
            delta = max(delta, abs(v - V[s]))
        if delta < theta:
            break
    return V

optimal_values = value_iteration(P, states, actions, gamma, theta)
print("Optimal Value Function:")
for s in states:
    print(f"V({s}) = {optimal_values[s]:.2f}")
    
