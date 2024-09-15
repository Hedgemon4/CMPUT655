import gymnasium
import numpy as np
import matplotlib.pyplot as plt

from assignments.assignment2.assignment2 import THETA

env = gymnasium.make("Gym-Gridworlds/Penalty-3x3-v0")

n_states = env.observation_space.n
n_actions = env.action_space.n

R = np.zeros((n_states, n_actions))
P = np.zeros((n_states, n_actions, n_states))
T = np.zeros((n_states, n_actions))

# THETA = 0.00001
THETA = 0.1
# THETA = 0.5

# define the optimal policy
pi_opt = np.zeros((n_states, n_actions))
pi_opt[0, 1] = 1.0
pi_opt[1, 2] = 1.0
pi_opt[4, 2] = 1.0
pi_opt[3, 1] = 1.0
pi_opt[6, 2] = 1.0
pi_opt[7, 2] = 1.0
pi_opt[8, 3] = 1.0
pi_opt[5, 3] = 1.0
pi_opt[2, 4] = 1.0

def compute_matrices():
    env.reset()
    for s in range(n_states):
        for a in range(n_actions):
            env.set_state(s)
            s_next, r, terminated, _, _ = env.step(a)
            R[s, a] = r
            P[s, a, s_next] = 1.0
            T[s, a] = terminated

def generalized_policy_iteration(**kwargs):
    initial_value = kwargs['initial_value']
    gamma = 0.99
    qk = np.full((n_states, n_actions), initial_value)
    pi = np.full((n_states, n_actions), 0.2)


    policy_stable = False
    bellman_errors = []
    while not policy_stable:
        # Policy evaluation
        policy_evaluation(gamma = gamma, pi = pi, qk = qk, bellman_errors = bellman_errors, cutoff=5)
        pi, policy_stable = policy_improvement_2(pi = pi, qk = qk)
    return pi, len(bellman_errors), bellman_errors


def policy_evaluation(**kwargs):
    gamma = kwargs['gamma']
    pi = kwargs['pi']
    qk = kwargs['qk']
    bellman_errors = kwargs['bellman_errors']
    cutoff = kwargs.get('cutoff', 0)

    delta = THETA + 1
    iterations = 0
    while delta >= THETA:
        if cutoff > 0 and iterations >= cutoff:
            break
        delta = 0
        q_old = qk.copy()
        for state in range(n_states):
            for action in range(n_actions):
                value = 0
                initial_q = qk[state, action]
                for state_prime in range(n_states):
                    dynamics_probability = P[state, action, state_prime]
                    reward = R[state, action]

                    inner_sum = 0
                    if T[state, action] == 0:
                        for action_prime in range(n_actions):
                            action_prob = pi[state_prime, action_prime]
                            inner_sum += action_prob * qk[state_prime, action_prime]
                    value += dynamics_probability * (reward + (gamma * inner_sum))
                qk[state, action] = value
                delta = max(delta, abs(initial_q - value))

        # log errors
        error = 0
        for state in range(n_states):
            for action in range(n_actions):
                error += abs(qk[state, action] - q_old[state, action])
        bellman_errors.append(error)
        iterations += 1

def policy_improvement(**kwargs):
    pi = kwargs['pi']
    qk = kwargs['qk']

    policy_stable = False
    improved_policy = np.zeros((n_states, n_actions))
    for state in range(n_states):
        best_action = 0
        max_value = 0
        for action in range(n_actions):
            value = qk[state, action]
            if value > max_value or action == 0:
                max_value = value
                best_action = action
        improved_policy[state, best_action] = 1.0

    if np.array_equal(pi, improved_policy):
        policy_stable = True
    return improved_policy, policy_stable


def policy_improvement_2(**kwargs):
    pi = kwargs['pi']
    qk = kwargs['qk']

    policy_stable = False
    policy_deterministic = True
    improved_policy = np.zeros((n_states, n_actions))
    for state in range(n_states):
        values = []
        for action in range(n_actions):
            values.append(qk[state, action])

        max_value = max(values)

        indicies_of_max = [index for index, value in enumerate(values) if value == max_value]
        probability = 1.0 / len(indicies_of_max)

        if len(indicies_of_max) > 1:
            policy_deterministic = False

        for index in indicies_of_max:
            improved_policy[state, index] = probability

    if np.array_equal(pi, improved_policy) and policy_deterministic:
        policy_stable = True
    return improved_policy, policy_stable



def policy_iteration(**kwargs):
    initial_value = kwargs['initial_value']
    gamma = 0.99
    qk = np.full((n_states, n_actions), initial_value)
    pi = np.full((n_states, n_actions), 0.2)


    policy_stable = False
    bellman_errors = []
    while not policy_stable:
        # Policy evaluation
        policy_evaluation(gamma = gamma, pi = pi, qk = qk, bellman_errors = bellman_errors)
        pi, policy_stable = policy_improvement_2(pi = pi, qk = qk)
    return pi, len(bellman_errors), bellman_errors


def value_iteration(**kwargs):
    initial_value = kwargs['initial_value']
    gamma = 0.99
    qk = np.full((n_states, n_actions), initial_value)


    bellman_errors = []
    delta = THETA + 1
    iterations = 0
    while delta >= THETA:
        delta = 0
        q_old = qk.copy()
        for state in range(n_states):
            for action in range(n_actions):
                value = 0
                initial_q = qk[state, action]
                for state_prime in range(n_states):
                    dynamics_probability = P[state, action, state_prime]
                    reward = R[state, action]

                    max_inner = 0
                    if T[state, action] == 0:
                        for action_prime in range(n_actions):
                            value = qk[state_prime, action_prime]
                            if action_prime == 0 or value > max_inner:
                                max_inner = value
                    value += dynamics_probability * (reward + (gamma * max_inner))
                qk[state, action] = value
                delta = max(delta, abs(initial_q - value))

        # log errors
        error = 0
        for state in range(n_states):
            for action in range(n_actions):
                error += abs(qk[state, action] - q_old[state, action])
        bellman_errors.append(error)
        iterations += 1

    improved_policy = np.zeros((n_states, n_actions))
    for state in range(n_states):
        best_action = 0
        max_value = 0
        for action in range(n_actions):
            value = qk[state, action]
            if value > max_value or action == 0:
                max_value = value
                best_action = action
        improved_policy[state, best_action] = 1.0

    return improved_policy, len(bellman_errors), bellman_errors


def plot_graphs():
    fig, axs = plt.subplots(3, 7)
    tot_iter_table = np.zeros((3, 7))
    gamma = 0.99
    initial_values = [-100.0, -10.0, -5.0, 0.0, 5.0, 10.0, 100.0]
    # initial_values = [5]
    for i, init_value in enumerate(initial_values):
        axs[0][i].set_title(f"$V_0$ = {init_value}")

        # VI
        # pi, tot_iter, be = value_iteration(gamma=gamma, initial_value=init_value)
        # tot_iter_table[0, i] = tot_iter
        # assert np.allclose(pi, pi_opt)
        # axs[0][i].plot(range(len(be)), be)

        # PI
        pi = np.full((n_states, n_actions), 0.2)
        pi, tot_iter, be = policy_iteration(gamma=gamma, policy=pi, initial_value=init_value)
        # pi = policy_iteration(gamma=gamma, policy=pi, initial_value=init_value)
        tot_iter_table[1, i] = tot_iter
        assert np.allclose(pi, pi_opt)
        print(np.allclose(pi, pi_opt))
        axs[1][i].plot(range(len(be)), be)

        # GPI
        pi = np.full((n_states, n_actions), 0.2)
        pi, tot_iter, be = generalized_policy_iteration(gamma=gamma, policy=pi, initial_value=init_value)
        tot_iter_table[2, i] = tot_iter
        assert np.allclose(pi, pi_opt)
        axs[2][i].plot(range(len(be)), be)

        if i == 0:
            axs[0][i].set_ylabel("VI")
            axs[1][i].set_ylabel("PI")
            axs[2][i].set_ylabel("GPI")

    plt.show()

    print(tot_iter_table.mean(-1))
    print(tot_iter_table.std(-1))


if __name__ == "__main__":
    compute_matrices()

    plot_graphs()
