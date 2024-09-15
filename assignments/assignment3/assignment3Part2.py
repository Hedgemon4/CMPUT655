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

# def policy_improvement(**kwargs):
#     policy = kwargs['policy']
#     qk = kwargs['qk']
#     improved_policy = np.zeros(policy.shape)
#     for state in range(n_states):
#         max_value = 0
#         best_action = 0
#         for action in range(n_actions):
#             value = qk[state, action]
#             if action == 0 or value > max_value:
#                 max_value = value
#                 best_action = action
#         improved_policy[state, best_action] = 1.0
#     policy_stable = np.array_equal(improved_policy, policy)
#     return {"isStable": policy_stable, "betterPolicy": improved_policy}

# def policy_improvement(**kwargs):
#     policy = kwargs['policy']
#     qk = kwargs['qk']
#     improved_policy = np.zeros(policy.shape)
#     deterministic = True
#     for state in range(n_states):
#         values = []
#         for action in range(n_actions):
#             values.append(qk[state, action])
#         max_value = max(values)
#         indicies_of_max = [index for index, value in enumerate(values) if value == max_value]
#         probability = 1.0 / len(indicies_of_max)
#
#         if len(indicies_of_max) > 1:
#             deterministic = False
#
#         for index in indicies_of_max:
#             improved_policy[state, index] = probability
#     policy_stable = np.array_equal(improved_policy, policy) and deterministic
#     return {"isStable": policy_stable, "betterPolicy": improved_policy}

# def policy_improvement(**kwargs):
#     policy = kwargs['policy']
#     gamma = kwargs['gamma']
#     qk = kwargs['qk']
#     improved_policy = np.zeros(policy.shape)
#     policy_stable = False

    # Improve policy
    # policy_stable = False
    # for state in range(n_states):
    #     max_value = 0
    #     best_action = 0
    #     for action in range(n_actions):
    #         value = 0
    #         for state_prime in range(n_states):
    #             dynamics_prob = P[state, action, state_prime]
    #             if dynamics_prob == 0:
    #                 continue
    #             reward = R[state, action]
    #             value_of_s_prime = vs[state_prime]
    #             if T[state, action] == 1:
    #                 value_of_s_prime = 0
    #             value += dynamics_prob * (reward + (gamma * value_of_s_prime))
    #         if action == 0:
    #             max_value = value
    #         elif value >= max_value:
    #             best_action = action
    #             max_value = value
    #
    #     improved_policy[state, best_action] = 1.0

    # for state in range(n_states):
    #     max_value = 0
    #     best_action = 0
    #     for action in range(n_actions):
    #         value = 0
    #         for state_prime in range(n_states):
    #             dynamics_prob = P[state, action, state_prime]
    #             if dynamics_prob == 0:
    #                 continue
    #             reward = R[state, action]
    #             inner_sum = 0
    #             if T[state, action] != 1:
    #                 for action_prime in range(n_actions):
    #                     action_prob = policy[state_prime, action_prime]
    #                     if action_prob == 0:
    #                         continue
    #                     inner_sum += action_prob * qk[state_prime, action_prime]
    #             value += dynamics_prob * (reward + (gamma * inner_sum))
    #         if action == 0:
    #             max_value = value
    #         elif value >= max_value:
    #             best_action = action
    #             max_value = value
    #     improved_policy[state, best_action] = 1.0
    # if np.array_equal(improved_policy, policy):
    #     policy_stable = True
    # return {"isStable": policy_stable, "betterPolicy": improved_policy}
    #

# def policy_improvement(**kwargs):

# def policy_evaluation(**kwargs):
#     gamma = kwargs.get("gamma", 1)
#     initial_value = kwargs.get("initial_value", 0)
#     max_iterations = kwargs.get("max_iterations", 100000000)
#     policy = kwargs.get("policy", np.zeros((n_states, n_actions)))
#     bellman_errors = kwargs.get("bellman_errors", [])
#     qk = kwargs.get("qk", np.zeros((n_states, n_actions)))
#     qk1 = qk.copy()
#
#     delta = THETA + 1
#
#     iterations = 0
#     while delta > THETA and iterations < max_iterations:
#         delta = 0
#         for state in range(n_states):
#             for action in range(n_actions):
#                 value = 0
#                 for state_prime in range(n_states):
#                     dynamics_prob = P[state, action, state_prime]
#                     # if dynamics_prob == 0:
#                     #     continue
#                     reward = R[state, action]
#                     inner_sum = 0
#                     if T[state, action] != 1:
#                         for action_prime in range(n_actions):
#                             action_prob = policy[state_prime, action_prime]
#                             # if action_prob == 0:
#                             #     continue
#                             inner_sum += action_prob * qk[state_prime, action_prime]
#                     value += dynamics_prob * (reward + (gamma * inner_sum))
#                 qk1[state, action] = value
#                 delta = max(delta, abs(qk[state, action] - qk1[state, action]))
#         error = 0
#         for state in range(n_states):
#             for action in range(n_actions):
#                 error += abs(qk1[state, action] - qk[state, action])
#         bellman_errors.append(error)
#         for state in range(n_states):
#             for action in range(n_actions):
#                 qk[state, action] = qk1[state, action]
#         iterations += 1
#     return {"values": qk1, "bellman_errors": bellman_errors}

# def policy_evaluation(**kwargs):
#     gamma = kwargs['gamma']
#     policy = kwargs['policy']
#     qk = kwargs['qk']
#     bellman_errors = kwargs['bellman_errors']
#
#     qk1 = qk.copy()
#     for state in range(n_states):
#         for action in range(n_actions):
#             dyn


# def policy_iteration(**kwargs):
#     # Initialization
#     gamma = kwargs.get("gamma", 1)
#     policy = kwargs.get("policy", np.full((n_states, n_actions), 0.2))
#     initial_value = kwargs.get("initial_value", 0)
#     qk = np.full((n_states, n_actions), initial_value)
#
#     policy_stable = False
#     bellman_errors = []
#     while not policy_stable:
#         # Policy Evaluation
#         qk = policy_evaluation(gamma=gamma, policy=policy, qk=qk, bellman_errors=bellman_errors)["values"]
#         # Policy Improvement
#         policy_stable, policy = policy_improvement(gamma=gamma, policy=policy, qk=qk).values()
#     return policy, len(bellman_errors), bellman_errors

def generalized_policy_iteration(**kwargs):
    return

# def value_iteration(**kwargs):
#     # Setup
#     gamma = kwargs.get("gamma", 1)
#     initial_value = kwargs.get("initial_value", 0)
#
#     # Iteration loop
#     bellman_errors = []
#     total_iterations = 0
#     qk = np.full((n_states, n_actions), initial_value)
#     qk1 = np.full((n_states, n_actions), initial_value)
#     delta = THETA + 1
#
#     # loop for every state
#     while delta > THETA:
#         delta = 0
#         for state in range(n_states):
#             for action in range(n_actions):
#                 for state_prime in range(n_states):
#                     dynamics_prob = P[state, action, state_prime]
#                     if dynamics_prob == 0:
#                         continue
#                     reward = R[state, action]
#                     inner_sum = 0
#                     if T[state, action] != 1:
#
#                     value += dynamics_prob * (reward + (gamma * value_of_s_prime))
#                 if action == 0 or value >= max_value:
#                     max_value = value
#             vk1[state] = max_value
#             delta = max(delta, abs(vk[state] - vk1[state]))
#         error = 0
#         for vs, vs1 in zip(vk, vk1):
#             error += abs(vs - vs1)
#         bellman_errors.append(error)
#         # copy over to vk array
#         for state in range(n_states):
#             vk[state] = vk1[state]
#         total_iterations += 1
#
#     # Now we need to derive the optimal policy from v*
#     policy = np.full((n_states, n_actions), 0.0)
#     for state in range(n_states):
#         max_value = 0
#         best_action = 0
#
#         for action in range(n_actions):
#             value = 0
#             for state_prime in range(n_states):
#                 dynamics_prob = P[state, action, state_prime]
#                 if dynamics_prob  == 0:
#                     continue
#                 reward = R[state, action]
#                 value_of_s_prime = vk[state_prime]
#                 if T[state, action] == 1:
#                     value_of_s_prime = 0
#                 value += dynamics_prob * (reward + (gamma * value_of_s_prime))
#             if action == 0:
#                 max_value = value
#             elif value >= max_value:
#                 best_action = action
#                 max_value = value
#         policy[state, best_action] = 1.0
#     return policy, total_iterations, bellman_errors
#

def policy_evaluation(**kwargs):
    gamma = kwargs['gamma']
    pi = kwargs['pi']
    qk = kwargs['qk']

    delta = THETA + 1
    while delta >= THETA:
        delta = 0
        for state in range(n_states):
            for action in range(n_actions):
                value = 0
                initial_q = qk[state, action]
                for state_prime in range(n_states):
                    dynamics_probability = P[state, action, state_prime]
                    reward = R[state, action]

                    inner_sum = 0
                    for action_prime in range(n_actions):
                        action_prob = pi[state_prime, action_prime]
                        inner_sum += action_prob * qk[state_prime, action_prime]
                    value += dynamics_probability * (reward + (gamma * inner_sum))
                qk[state, action] = value
                delta = max(delta, abs(initial_q - value))

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



def policy_iteration(**kwargs):
    initial_value = 0.0
    gamma = 0.99
    qk = np.full((n_states, n_actions), initial_value)
    pi = np.full((n_states, n_actions), 0.2)


    policy_stable = False
    while not policy_stable:
        # Policy evaluation
        policy_evaluation(gamma = gamma, pi = pi, qk = qk)
        pi, policy_stable = policy_improvement(pi = pi, qk = qk)
    return pi


def plot_graphs():
    fig, axs = plt.subplots(3, 7)
    tot_iter_table = np.zeros((3, 7))
    gamma = 0.99
    initial_values = [-100, -10, -5, 0, 5, 10, 100]
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
        # pi, tot_iter, be = policy_iteration(gamma=gamma, policy=pi, initial_value=init_value)
        pi = policy_iteration(gamma=gamma, policy=pi, initial_value=init_value)
        # tot_iter_table[1, i] = tot_iter
        # assert np.allclose(pi, pi_opt)
        print(np.allclose(pi, pi_opt))
        # axs[1][i].plot(range(len(be)), be)

        # GPI
        # pi = np.full((n_states, n_actions), 0.2)
        # pi, tot_iter, be = generalized_policy_iteration(gamma=gamma, policy=pi, initial_value=init_value)
        # tot_iter_table[2, i] = tot_iter
        # assert np.allclose(pi, pi_opt)
        # axs[2][i].plot(range(len(be)), be)

        # if i == 0:
        #     axs[0][i].set_ylabel("VI")
        #     axs[1][i].set_ylabel("PI")
        #     axs[2][i].set_ylabel("GPI")

    plt.show()

    print(tot_iter_table.mean(-1))
    print(tot_iter_table.std(-1))


if __name__ == "__main__":
    compute_matrices()

    plot_graphs()
