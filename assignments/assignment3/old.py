import gymnasium
import numpy as np
import matplotlib.pyplot as plt

env = gymnasium.make("Gym-Gridworlds/Penalty-3x3-v0")

n_states = env.observation_space.n
n_actions = env.action_space.n

R = np.zeros((n_states, n_actions))
P = np.zeros((n_states, n_actions, n_states))
T = np.zeros((n_states, n_actions))

THETA = 0.05

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


def policy_improvement(**kwargs):
    # Get stuff from kwargs
    policy = kwargs["policy"]
    gamma = kwargs["gamma"]
    vs = kwargs["vs"]

    improved_policy = np.zeros(policy.shape)

    # Improve policy
    policy_stable = False
    for state in range(n_states):
        max_value = 0
        best_action = 0

        for action in range(n_actions):
            value = 0
            for state_prime in range(n_states):
                dynamics_prob = P[state, action, state_prime]
                if dynamics_prob == 0:
                    continue
                reward = R[state, action]
                value_of_s_prime = vs[state_prime]
                if T[state, action] == 1:
                    value_of_s_prime = 0
                value += dynamics_prob * (reward + (gamma * value_of_s_prime))
            if action == 0:
                max_value = value
            elif value > max_value:
                best_action = action
                max_value = value

        improved_policy[state, best_action] = 1.0

    if np.array_equal(improved_policy, policy):
        policy_stable = True
    return {"isStable": policy_stable, "policy": improved_policy}


def policy_evaluation(**kwargs):
    # Get the values we need from good ol kwargs
    gamma = kwargs.get("gamma", 1)
    policy = kwargs.get("policy", np.zeros((n_states, n_actions)))
    bellman_errors = kwargs.get("bellman_errors", [])
    vk = kwargs.get("vk", [0] * n_states)
    vk1 = [item for item in vk]

    delta = THETA + 1

    # Loop to get best value function
    while delta > THETA:
        delta = 0
        for state in range(n_states):
            # perform our bellman update on vk1
            max_value = 0
            for action in range(n_actions):
                value = 0
                for state_prime in range(n_states):
                    dynamics_prob = P[state, action, state_prime]
                    if dynamics_prob == 0:
                        continue
                    reward = R[state, action]
                    value_of_s_prime = vs[state_prime]
                    if T[state, action] == 1:
                        value_of_s_prime = 0
                    value += dynamics_prob * (reward + (gamma * value_of_s_prime))
                if action == 0 or value > max_value:
                    max_value = value
            vk1[state] = max_value
            delta = max(delta, abs(vk[state] - vk1[state]))
        error = 0
        for vs, vs1 in zip(vk, vk1):
            error += abs(vs - vs1)
        bellman_errors.append(error)
        # copy over to vk array
        for state in range(n_states):
            vk[state] = vk1[state]

    return {"values": vk, "bellman_errors": bellman_errors}


def policy_iteration(**kwargs):
    # Initialization
    gamma = kwargs.get("gamma", 1)
    policy = kwargs.get("policy", np.full((n_states, n_actions), 0.2))
    initial_value = kwargs.get("initial_value", 0)
    vs = [initial_value] * n_states

    policy_stable = False
    total_iterations = 0
    bellman_errors = []
    while not policy_stable:
        # Policy Evaluation
        vs = policy_evaluation(
            gamma=gamma, policy=policy, vk=vs, bellman_errors=bellman_errors
        )["values"]
        # Policy Improvement
        policy_stable, policy = policy_improvement(
            gamma=gamma, policy=policy, vs=vs
        ).values()
        total_iterations += 1
    return policy, total_iterations, bellman_errors


def generalized_policy_iteration(**kwargs):
    return


def value_iteration(**kwargs):
    # Setup
    gamma = kwargs.get("gamma", 1)
    policy = kwargs.get("policy", np.full((n_states, n_actions), 0.2))
    initial_value = kwargs.get("initial_value", 0)

    # Iteration loop
    bellman_errors = []
    vk = [initial_value] * n_states
    vk1 = [initial_value] * n_states
    total_iterations = 0

    delta = THETA + 1

    # loop for every state
    while delta > THETA:
        delta = 0
        for state in range(n_states):
            # perform our bellman update on vk1
            value = 0
            for action in range(n_actions):
                # probability of choosing this action with our policy (this is the sum)
                action_prob = policy[state, action]
                if action_prob == 0:
                    continue
                # otherwise
                for state_prime in range(n_states):
                    dynamics_prob = P[state, action, state_prime]
                    if dynamics_prob == 0:
                        continue
                    # Now we can finally just do our value update
                    reward = R[state, action]
                    value_of_s_prime = vk[state_prime]
                    if T[state, action] == 1:
                        value_of_s_prime = 0
                    value += (
                        action_prob
                        * dynamics_prob
                        * (reward + (gamma * value_of_s_prime))
                    )
            vk1[state] = value
            delta = max(delta, abs(vk[state] - vk1[state]))
        error = 0
        for vs, vs1 in zip(vk, vk1):
            error += abs(vs - vs1)
        bellman_errors.append(error)
        # copy over to vk array
        for state in range(n_states):
            vk[state] = vk1[state]

    return


def plot_graphs():
    fig, axs = plt.subplots(3, 7)
    tot_iter_table = np.zeros((3, 7))
    gamma = 0.99
    for i, init_value in enumerate([-100, -10, -5, 0, 5, 10, 100]):
        axs[0][i].set_title(f"$V_0$ = {init_value}")

        pi = np.full((n_states, n_actions), 0.2)

        # pi, tot_iter, be = value_iteration(...)
        # tot_iter_table[0, i] = tot_iter
        # assert np.allclose(pi, pi_opt)
        # axs[0][i].plot(...)
        #
        pi, tot_iter, be = policy_iteration(
            gamma=gamma, policy=pi, initial_value=init_value
        )
        tot_iter_table[1, i] = tot_iter
        assert np.allclose(pi, pi_opt)
        axs[1][i].plot(range(len(be)), be)
        #
        # pi, tot_iter, be = generalized_policy_iteration(...)
        # tot_iter_table[2, i] = tot_iter
        # assert np.allclose(pi, pi_opt)
        # axs[2][i].plot(...)

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
