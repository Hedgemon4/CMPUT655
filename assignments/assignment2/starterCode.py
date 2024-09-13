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


def bellman_v(**kwargs):
    # Get the values we need from good ol kwargs
    gamma = kwargs.get("gamma", 1)
    initial_value = kwargs.get("initial_value", 0)

    delta = THETA + 1

    # initialize value function approximation
    vk = [initial_value] * n_states
    vk1 = [initial_value] * n_states

    # initalize bellman error array
    bellman_errors = []

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
    return {"values": vk1, "bellman_errors": bellman_errors}


def bellman_q_old(**kwargs):
    # Get the values we need from good ol kwargs
    gamma = kwargs.get("gamma", 1)
    initial_value = kwargs.get("initial_value", 0)

    delta = THETA + 1

    # initialize q function approximations
    qk = np.full((n_states, n_actions), initial_value)
    qk1 = np.zeros((n_states, n_actions))

    bellman_errors = []

    while delta > THETA:
        delta = 0
        # Here we do state action pairs to start, so we immediately loop through all state action pairs
        for state in range(n_states):
            for action in range(n_actions):
                value = 0
                for state_prime in range(n_states):
                    dynamics_prob = P[state, action, state_prime]
                    if dynamics_prob == 0:
                        continue
                    reward = R[state, action]
                    inner_sum = 0
                    # Account for terminal state
                    if T[state, action] != 1:
                        for action_prime in range(n_actions):
                            action_prob = policy[state_prime, action_prime]
                            if action_prob == 0:
                                continue
                            inner_sum += (action_prob * qk[state_prime, action_prime])
                    value += dynamics_prob * (reward + (gamma * inner_sum))
                qk1[state, action] = value
                delta = max(delta, abs(qk[state, action] - qk1[state, action]))
        error = 0
        # Compute the bellman error for this iteration
        for state in range(n_states):
            for action in range(n_actions):
                error += abs(qk1[state, action] - qk[state, action])
        bellman_errors.append(error)
        # Update qk to be qk1
        for state in range(n_states):
            for action in range(n_actions):
                qk[state, action] = qk1[state, action]
    return {"values": qk1, "bellman_errors": bellman_errors}

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
                            inner_sum += (action_prob * qk[state_prime, action_prime])
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

def plot_graphs():
    gammas = [0.01, 0.5, 0.99]
    for init_value in [-10, 0, 10]:
        fig, axs = plt.subplots(2, len(gammas))
        fig.suptitle(f"$V_0$: {init_value}")
        for i, gamma in enumerate(gammas):
            results = bellman_v(gamma=gamma, initial_value=init_value)
            values = np.array(results["values"]).reshape(3, 3)
            errors = results["bellman_errors"]
            axs[0][i].imshow(values)
            axs[1][i].plot(range(len(errors)), errors)
            axs[0][i].set_title(f"$\gamma$ = {gamma}")

        fig, axs = plt.subplots(n_actions + 1, len(gammas))
        fig.suptitle(f"$Q_0$: {init_value}")
        for i, gamma in enumerate(gammas):
            results = bellman_q(gamma=gamma, initial_value=init_value)
            for a in range(n_actions):
                values = results["values"][:, a].reshape(3, 3)
                axs[a][i].imshow(values)
            errors = results["bellman_errors"]
            axs[-1][i].plot(range(len(errors)), errors)
            axs[0][i].set_title(f'$\gamma$ = {gamma}')

        plt.show()


if __name__ == "__main__":
    compute_matrices()

    # Setup our optimal
    policy[0, 1] = 1.0
    policy[1, 2] = 1.0
    policy[4, 2] = 1.0
    policy[3, 1] = 1.0
    policy[6, 2] = 1.0
    policy[7, 2] = 1.0
    policy[8, 3] = 1.0
    policy[5, 3] = 1.0
    policy[2, 4] = 1.0

    # plot things
    plot_graphs()
