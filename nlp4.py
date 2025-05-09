import gym
import numpy as np
from collections import defaultdict

env = gym.make("FrozenLake-v1", is_slippery=False)

gamma = 0.9
alpha = 0.1
lmbda = 0.8
episodes = 5000

V = defaultdict(float)

for _ in range(episodes):
    state = env.reset()
    eligibility = defaultdict(float)
    done = False

    while not done:
        action = env.action_space.sample()
        next_state, reward, done, _ = env.step(action)

        td_error = reward + gamma * V[next_state] - V[state]
        eligibility[state] += 1

        for s in V:
            V[s] += alpha * td_error * eligibility[s]
            eligibility[s] *= gamma * lmbda

        state = next_state

print("Estimated Value Function with TD(λ):")
for s in range(env.observation_space.n):
    print(f"V({s}) = {V[s]:.2f}")
    

import gym
import numpy as np

env = gym.make("CartPole-v1")
obs_space = env.observation_space.shape[0]
action_space = env.action_space.n

# Initialize policy parameters
theta = np.random.rand(obs_space, action_space)
gamma = 0.99
alpha = 0.01
episodes = 1000

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / np.sum(e_x)

for episode in range(episodes):
    state = env.reset()
    states, actions, rewards = [], [], []
    done = False

    while not done:
        probs = softmax(np.dot(state, theta))
        action = np.random.choice(action_space, p=probs)
        next_state, reward, done, _ = env.step(action)

        states.append(state)
        actions.append(action)
        rewards.append(reward)
        state = next_state

    # Compute returns
    G = 0
    returns = []
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)

    # Update policy
    for t in range(len(states)):
        s = states[t]
        a = actions[t]
        probs = softmax(np.dot(s, theta))
        grad_log = np.zeros_like(theta)
        for i in range(action_space):
            grad_log[:, i] = -probs[i] * s
        grad_log[:, a] += s
        theta += alpha * returns[t] * grad_log

print("REINFORCE training complete.")

import numpy as np
import random

floors = 5
actions = ['up', 'down', 'stay']
q_table = np.zeros((floors, len(actions)))

alpha = 0.1
gamma = 0.9
epsilon = 0.1
episodes = 1000

for _ in range(episodes):
    state = random.randint(0, floors - 1)
    target_floor = random.randint(0, floors - 1)
    
    while state != target_floor:
        if random.random() < epsilon:
            action = random.randint(0, len(actions) - 1)
        else:
            action = np.argmax(q_table[state])
        
        if actions[action] == 'up' and state < floors - 1:
            next_state = state + 1
        elif actions[action] == 'down' and state > 0:
            next_state = state - 1
        else:
            next_state = state

        reward = 1 if next_state == target_floor else -0.1

        q_table[state][action] += alpha * (reward + gamma * np.max(q_table[next_state]) - q_table[state][action])
        state = next_state

print("Learned Q-table (Elevator Problem):")
for f in range(floors):
    print(f"Floor {f}: {q_table[f]}")



