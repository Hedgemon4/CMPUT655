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

    # We can just set an arbitrary algorithm parameter here because I can and Adrian can't stop me
    THETA = 0.001
    delta = 0.002

    # initialize value function approximation
    vk = [initial_value] * n_states
    vk1 = [0] * n_states

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
                    reward = R[state_prime, action]
                    value += action_prob * dynamics_prob * (reward + (gamma * vk[state_prime]))
            vk1[state] = value
            delta = max(delta, abs(vk[state] - vk1[state]))
            print(delta)
        error = 0
        for vs, vs1 in zip(vk, vk1) :
            error += abs(vs - vs1)
        bellman_errors.append(error)
        # copy over to vk array
        for state in range(n_states):
            vk[state] = vk1[state]
    return {"values": vk1, "bellman_errors": bellman_errors}


# def bellman_q(**kwargs):
#     return
#
def plot_graphs():
    gammas = [0.01, 0.5, 0.99]
    for init_value in [-10, 0, 10]:
        fig, axs = plt.subplots(2, len(gammas))
        fig.suptitle(f"$V_0$: {init_value}")
        for i, gamma in enumerate(gammas):
            results = bellman_v(gamma=gamma, initial_value=init_value)
            axs[0][i].imshow(results.get("values"))
            axs[1][i].plot(results.get("bellman_errors"))
            axs[0][i].set_title(f'$\gamma$ = {gamma}')

        # fig, axs = plt.subplots(n_actions + 1, len(gammas))
        # fig.suptitle(f"$Q_0$: {init_value}")
        # for i, gamma in enumerate(gammas):
        #     ... = bellman_q(...)
        #     for a in range(n_actions):
        #         axs[a][i].imshow(...)
        #     axs[-1][i].plot(...)
        #     axs[0][i].set_title(f'$\gamma$ = {gamma}')

        plt.show()

if __name__ == '__main__':
    compute_matrices()

    # Setup our optimal
    policy[0, 1] = 1.0
    policy[3, 1] = 1.0
    policy[6, 2] = 1.0
    policy[7, 2] = 1.0
    policy[8, 3] = 1.0
    policy[5, 3] = 1.0
    policy[2, 4] = 1.0

    # plot things
    plot_graphs()