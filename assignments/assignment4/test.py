import gymnasium
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=3)

env = gymnasium.make("Gym-Gridworlds/Penalty-3x3-v0")
n_states = env.observation_space.n
n_actions = env.action_space.n

R = np.zeros((n_states, n_actions))
P = np.zeros((n_states, n_actions, n_states))
T = np.zeros((n_states, n_actions))

env.reset()
for s in range(n_states):
    for a in range(n_actions):
        env.unwrapped.set_state(s)
        s_next, r, terminated, _, _ = env.step(a)
        R[s, a] = r
        P[s, a, s_next] = 1.0
        T[s, a] = terminated

P = P * (1.0 - T[..., None])  # next state probability for terminal transitions is 0


def bellman_q(pi, gamma):
    I = np.eye(n_states * n_actions)
    P_under_pi = (P[..., None] * pi[None, None]).reshape(
        n_states * n_actions, n_states * n_actions
    )
    return (
        (R.ravel() * np.linalg.inv(I - gamma * P_under_pi))
        .sum(-1)
        .reshape(n_states, n_actions)
    )


def episode(env, Q, eps, seed):
    data = dict()
    data["s"] = []
    data["a"] = []
    data["r"] = []
    s, _ = env.reset(seed=int(seed))
    done = False
    while not done:
        a = eps_greedy_action(Q, s, eps)
        s_next, r, terminated, truncated, _ = env.step(a)
        done = terminated or truncated
        data["s"].append(s)
        data["a"].append(a)
        data["r"].append(r)
        s = s_next
    return data


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


def monte_carlo(
    env, Q, gamma, eps_decay, max_steps, episodes_per_iteration, use_is
):
    # Initialize Values we Need
    eps = 1
    total_steps = 0
    last_update_step = 0
    bellman_errors = []
    C = np.zeros((n_states, n_actions))

    # Get initial error
    error = bellman_error(Q, eps, gamma)

    while total_steps < max_steps:
        # Generate Episodes
        episodes = []
        for i in range(episodes_per_iteration):
            data = episode(env, Q, eps, seed)
            episodes.append(data)
            episode_steps = len(data["s"])
            total_steps += episode_steps
            # decay epsilon
            eps = max(eps - eps_decay / max_steps * episode_steps, 0.01)

        # Update our Q function
        for item in episodes:
            G = 0
            W = 1
            # for i in reversed(range(len(item["s"]))):
            #     G = gamma * G + item["r"][i]
            # for s, a, r in zip(reversed(item["s"]), reversed(item["a"]), reversed(item["r"])):
            #     G = gamma * G + r
            # Not sure about the t + 1
            for i in reversed(range(len(item["s"]) - 1)):
                a = item["a"][i]
                s = item["s"][i]
                G = gamma * G + item["r"][i + 1]
                C[s, a] += W
                Q[s, a] += (W / C[s, a]) * (G - Q[s, a])

        # Add our previous bellman error
        bellman_errors[last_update_step:total_steps] = [error] * (
            total_steps - last_update_step
        )
        last_update_step = total_steps
        error = bellman_error(Q, eps, gamma)

    return Q, bellman_errors


def bellman_error(Q, eps, gamma):
    pi = eps_greedy_probs(Q, eps)
    expected_q = bellman_q(pi, gamma)
    error = 0
    for state in range(n_states):
        for action in range(n_actions):
            error += abs(Q[state, action] - expected_q[state, action])
    return error


def error_shade_plot(ax, data, stepsize, **kwargs):
    y = np.nanmean(data, 0)
    x = np.arange(len(y))
    x = [stepsize * step for step in range(len(y))]
    (line,) = ax.plot(x, y, **kwargs)
    error = np.nanstd(data, axis=0)
    error = 1.96 * error / np.sqrt(data.shape[0])
    ax.fill_between(
        x, y - error, y + error, alpha=0.2, linewidth=0.0, color=line.get_color()
    )


init_value = 0.0
gamma = 0.9
max_steps = 2000
horizon = 10

episodes_per_iteration = [1, 10, 50]
decays = [1, 2, 5]
seeds = np.arange(10)

results = np.zeros(
    (
        len(episodes_per_iteration),
        len(decays),
        len(seeds),
        max_steps,
    )
)

fig, axs = plt.subplots(1, 2)
plt.ion()
plt.show()

use_is = False  # repeat with True
for ax, reward_noise_std in zip(axs, [0.0, 3.0]):
    ax.set_prop_cycle(
        color=[
            "red",
            "green",
            "blue",
            "black",
            "orange",
            "cyan",
            "brown",
            "gray",
            "pink",
        ]
    )
    ax.set_xlabel("Steps")
    ax.set_ylabel("Absolute Bellman Error")
    env = gymnasium.make(
        "Gym-Gridworlds/Penalty-3x3-v0",
        max_episode_steps=horizon,
        reward_noise_std=reward_noise_std,
    )
    for j, episodes in enumerate(episodes_per_iteration):
        for k, decay in enumerate(decays):
            for seed in seeds:
                np.random.seed(seed)
                Q = np.zeros((n_states, n_actions)) + init_value
                Q, be = monte_carlo(
                    env, Q, gamma, decay / max_steps, max_steps, episodes, False
                )
                results[j, k, seed] = be[0:2000]
            error_shade_plot(
                ax,
                results[j, k],
                stepsize=1,
                label=f"Episodes: {episodes}, Decay: {decay}",
            )
            ax.legend()
            # plt.draw()
            # plt.pause(0.001)

plt.ioff()
plt.show()
