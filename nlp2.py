import numpy as np
import random

# Define a simple k-armed bandit problem
k = 5
true_action_values = np.random.normal(0, 1, k)  # True reward for each arm

# Hyperparameters
epsilon = 0.1
steps = 1000

# Initialize estimates and counts
action_values = np.zeros(k)
action_counts = np.zeros(k)

rewards = []

for step in range(steps):
    # Exploration vs. Exploitation
    if random.random() < epsilon:
        action = np.random.randint(k)  # Explore
    else:
        action = np.argmax(action_values)  # Exploit

    # Simulate reward (true mean + noise)
    reward = np.random.binomial(n=1, p=1 / (1 + np.exp(-true_action_values[action])))
    rewards.append(reward)

    # Update counts and estimated value
    action_counts[action] += 1
    action_values[action] += (reward - action_values[action]) / action_counts[action]

print("Estimated Action Values:", action_values)
print("True Action Values:", true_action_values)
print("Total Reward:", sum(rewards))


import numpy as np
import math

k = 5
true_action_probs = np.random.uniform(0.1, 0.9, k)  # Prob of reward for each arm

steps = 1000
action_values = np.zeros(k)
action_counts = np.zeros(k)

rewards = []

for t in range(1, steps + 1):
    ucb_values = np.zeros(k)
    for i in range(k):
        if action_counts[i] == 0:
            ucb_values[i] = float('inf')  # Ensure every action is taken at least once
        else:
            bonus = math.sqrt((2 * math.log(t)) / action_counts[i])
            ucb_values[i] = action_values[i] + bonus

    action = np.argmax(ucb_values)

    reward = np.random.binomial(1, true_action_probs[action])
    rewards.append(reward)

    action_counts[action] += 1
    action_values[action] += (reward - action_values[action]) / action_counts[action]

print("Estimated Action Values:", action_values)
print("True Action Probabilities:", true_action_probs)
print("Total Reward:", sum(rewards))


