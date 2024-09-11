import gymnasium
import numpy as np
import matplotlib.pyplot as plt

env = gymnasium.make("Gym-Gridworlds/Penalty-3x3-v0")

n_states = env.observation_space.n
n_actions = env.action_space.n

R = np.zeros((n_states, n_actions))
P = np.zeros((n_states, n_actions, n_states))
T = np.zeros((n_states, n_actions))

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
    delta = 1

    # Make a dummy policy for now so my smoll brain can figure this out
    pi = np.zeros((n_states, n_actions))

    # initialize value function approximation
    vk = [initial_value] * n_states
    vk1 = [initial_value] * n_states

    # loop for every state
    while delta > THETA:
        for state in range(n_states):
            # perform our bellman update on vk1
            value = 0
            for action in range(n_actions):
                # probability of choosing this action with our policy (this is the sum)
                action_prob = pi[state, action]
                if action_prob == 0:
                    continue
                # otherwise
                for state_prime in range(n_states):
                    dynamics_prob = P[state, action, state_prime]
                    if dynamics_prob == 0:
                        continue
                    # Now we can finally just do our value update
                    reward = R[state_prime, action]
                    value += action_prob * dynamics_prob (reward + (gamma * vk[state_prime]))


    return 0

# def bellman_q(**kwargs):
#     return
#
# gammas = [0.01, 0.5, 0.99]
# for init_value in [-10, 0, 10]:
#     fig, axs = plt.subplots(2, len(gammas))
#     fig.suptitle(f"$V_0$: {init_value}")
#     for i, gamma in enumerate(gammas):
#         ... = bellman_v(...)
#         axs[0][i].imshow(...)
#         axs[1][i].plot(...)
#         axs[0][i].set_title(f'$\gamma$ = {gamma}')
#
#     fig, axs = plt.subplots(n_actions + 1, len(gammas))
#     fig.suptitle(f"$Q_0$: {init_value}")
#     for i, gamma in enumerate(gammas):
#         ... = bellman_q(...)
#         for a in range(n_actions):
#             axs[a][i].imshow(...)
#         axs[-1][i].plot(...)
#         axs[0][i].set_title(f'$\gamma$ = {gamma}')
#
#     plt.show()
