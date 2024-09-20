import numpy as np
n_states = 9
n_actions = 5

def eps_greedy_probs(Q, eps):
    max_actions = np.argmax(Q, axis=1)
    probabilities = np.full((n_states, n_actions), eps / n_actions, dtype=float)
    for state in range(n_states):
        probabilities[state, max_actions[state]] += 1 - eps
    return probabilities


def eps_greedy_action(Q, s, eps):
    probs = eps_greedy_probs(Q, eps)
    choices = list(range(n_actions))
    return np.random.choice(choices, p=probs[s])

Q = np.full((n_states, n_actions), 0.8, dtype=float)
print(eps_greedy_probs(Q, 0.05))
