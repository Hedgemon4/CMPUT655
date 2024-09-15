import gymnasium
import numpy as np
import matplotlib.pyplot as plt

env = gymnasium.make("Gym-Gridworlds/Penalty-3x3-v0")

n_states = env.observation_space.n
n_actions = env.action_space.n

R = np.zeros((n_states, n_actions))
P = np.zeros((n_states, n_actions, n_states))
T = np.zeros((n_states, n_actions))

# setup what the policy is we want to run
policy = np.zeros((n_states, n_actions))

# We can just set an arbitrary algorithm parameter here because I can and Adrian can't stop me
THETA = 0.05


def compute_matrices():
    env.reset()
    for s in range(n_states):
        for a in range(n_actions):
            env.set_state(s)
            s_next, r, terminated, _, _ = env.step(a)
            R[s, a] = r
            P[s, a, s_next] = 1.0
            T[s, a] = terminated


def bellman_q(**kwargs):
    # Get the values we need from good ol kwargs
    gamma = kwargs.get("gamma", 1)
    initial_value = kwargs.get("initial_value", 0)

    delta = THETA + 1

    # initialize q function approximations
    qk = np.full((n_states, n_actions), initial_value)
    qk1 = np.full((n_states, n_actions), initial_value)

    bellman_errors = []

    while delta > THETA:
        delta = 0
        for state in range(n_states):
            for action in range(n_actions):
                value = 0
                for state_prime in range(n_states):
                    dynamics_prob = P[state, action, state_prime]
                    if dynamics_prob == 0:
                        continue
                    reward = R[state, action]
                    inner_sum = 0
                    if T[state, action] != 1:
                        for action_prime in range(n_actions):
                            action_prob = policy[state_prime, action_prime]
                            if action_prob == 0:
                                continue
                            inner_sum += action_prob * qk[state_prime, action_prime]
                    value += dynamics_prob * (reward + (gamma * inner_sum))
                qk1[state, action] = value
                delta = max(delta, abs(qk[state, action] - qk1[state, action]))
        error = 0
        for state in range(n_states):
            for action in range(n_actions):
                error += abs(qk1[state, action] - qk[state, action])
        bellman_errors.append(error)
        for state in range(n_states):
            for action in range(n_actions):
                qk[state, action] = qk1[state, action]
    return {"values": qk1, "bellman_errors": bellman_errors}
