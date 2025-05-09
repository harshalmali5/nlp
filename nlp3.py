import gym
import numpy as np
from collections import defaultdict

env = gym.make("FrozenLake-v1", is_slippery=False)  # Deterministic version

# Define a random policy
policy = defaultdict(lambda: env.action_space.sample())

# Initialize value function
V = defaultdict(float)
returns = defaultdict(list)
num_episodes = 10000
gamma = 0.9

for _ in range(num_episodes):
    state = env.reset()
    episode = []
    done = False

    while not done:
        action = policy[state]
        next_state, reward, done, _ = env.step(action)
        episode.append((state, reward))
        state = next_state

    G = 0
    visited_states = set()
    for state, reward in reversed(episode):
        G = gamma * G + reward
        if state not in visited_states:
            returns[state].append(G)
            V[state] = np.mean(returns[state])
            visited_states.add(state)

# Display estimated value function
print("Estimated Value Function:")
for s in range(env.observation_space.n):
    print(f"V({s}) = {V[s]:.2f}")
    
    
import gym
import numpy as np
from collections import defaultdict

env = gym.make("FrozenLake-v1", is_slippery=False)

alpha = 0.1      # Learning rate
gamma = 0.99     # Discount factor
epsilon = 0.1    # Exploration rate
num_episodes = 5000

Q = defaultdict(lambda: np.zeros(env.action_space.n))

def epsilon_greedy_policy(state, Q, epsilon):
    if np.random.rand() < epsilon:
        return env.action_space.sample()
    return np.argmax(Q[state])

for _ in range(num_episodes):
    state = env.reset()
    action = epsilon_greedy_policy(state, Q, epsilon)
    done = False

    while not done:
        next_state, reward, done, _ = env.step(action)
        next_action = epsilon_greedy_policy(next_state, Q, epsilon)
        
        td_target = reward + gamma * Q[next_state][next_action]
        td_error = td_target - Q[state][action]
        Q[state][action] += alpha * td_error

        state = next_state
        action = next_action

# Display learned Q-values
print("Learned Q-values using SARSA:")
for s in range(env.observation_space.n):
    print(f"State {s}: {Q[s]}")
    
    