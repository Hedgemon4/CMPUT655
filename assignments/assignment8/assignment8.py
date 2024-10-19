from enum import Enum

import gymnasium
from fontTools.misc.bezierTools import epsilon

import gym_gridworlds
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


class Baseline(Enum):
    NONE = "none"
    AVERAGE = "average"
    OPTIMAL = "optimal"


np.set_printoptions(precision=3, suppress=True)
USE_GRID_WORLD = False


# https://en.wikipedia.org/wiki/Pairing_function
def cantor_pairing(x, y):
    return int(0.5 * (x + y) * (x + y + 1) + y)


def rbf_features(x: np.array, c: np.array, s: np.array) -> np.array:
    return np.exp(-(((x[:, None] - c[None]) / s[None]) ** 2).sum(-1) / 2.0)


def expected_return(env, weights, gamma, episodes=100, eps=1.0):
    G = np.zeros(episodes)
    for e in range(episodes):
        s, _ = env.reset(seed=e)
        done = False
        t = 0
        while not done:
            phi = get_phi(s)
            if USE_GRID_WORLD:
                a = eps_greedy_action(phi, weights, 0)
                s_next, r, terminated, truncated, _ = env.step(a)
            else:
                a = np.dot(phi, weights)
                a_clip = np.clip(
                    a, env.action_space.low, env.action_space.high
                )  # this is for the Pendulum
                # a = softmax_action(phi, weights, eps)  # this is for the Gridworld
                s_next, r, terminated, truncated, _ = env.step(
                    a_clip
                )  # replace with a for Gridworld
            done = terminated or truncated
            G[e] += gamma**t * r
            s = s_next
            t += 1
    return G.mean()


def collect_data(env, weights, sigma, n_episodes):
    data = dict()
    data["phi"] = []
    data["a"] = []
    data["r"] = []
    data["done"] = []
    for ep in range(n_episodes):
        episode_seed = cantor_pairing(ep, seed)
        s, _ = env.reset(seed=episode_seed)
        done = False
        while not done:
            phi = get_phi(s)
            if USE_GRID_WORLD:
                a = softmax_action(phi, weights, 1.0)
                s_next, r, terminated, truncated, _ = env.step(a)
            else:
                a = gaussian_action(phi, weights, sigma)
                a_clip = np.clip(
                    a, env.action_space.low, env.action_space.high
                )  # only for Gaussian policy
                s_next, r, terminated, truncated, _ = env.step(a_clip)
            done = terminated or truncated
            data["phi"].append(phi)
            data["a"].append(a)
            data["r"].append(r)
            data["done"].append(terminated or truncated)
            s = s_next
    return data


def eps_greedy_action(phi, weights, eps):
    if np.random.rand() < eps:
        return np.random.randint(n_actions)
    else:
        Q = np.dot(phi, weights).ravel()
        best = np.argwhere(Q == Q.max())
        i = np.random.choice(range(best.shape[0]))
        return best[i][0]


def softmax_probs(phi, weights, eps):
    q = np.dot(phi, weights)
    # this is a trick to make it more stable
    # see https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
    q_exp = np.exp((q - np.max(q, -1, keepdims=True)) / max(eps, 1e-12))
    probs = q_exp / q_exp.sum(-1, keepdims=True)
    return probs


def softmax_action(phi, weights, eps):
    probs = softmax_probs(phi, weights, eps)
    return np.random.choice(weights.shape[1], p=probs.ravel())


def dlog_softmax_probs(phi, weights, eps, act):
    # implement log-derivative of pi
    phi_sa = phi[..., None].repeat(n_actions, axis=-1)
    # Make mask
    # TODO: How to make this mask with a loop instead
    mask = np.zeros((act.shape[0], phi.shape[1], n_actions))
    for i, num in enumerate(act[:, 0]):
        mask[i][:, num] = 1
    probs = softmax_probs(phi, weights, eps)
    return phi_sa * mask - phi_sa * probs[:, None]


def gaussian_action(phi: np.array, weights: np.array, sigma: np.array):
    mu = np.dot(phi, weights)
    return np.random.normal(mu, sigma**2)


def dlog_gaussian_probs(
    phi: np.array, weights: np.array, sigma: np.array, action: np.array
):
    # implement log-derivative of pi with respect to the mean only
    # diag_covar_inverse = np.linalg.inv(np.square(np.diag(np.full(action.shape[0], sigma))))
    # diag_covar_inverse = np.diag(np.full(action.shape[0], (1 / sigma) ** 2))
    return ((1 / sigma) ** 2) * (action - np.dot(phi, weights))[:, None] * phi[..., None]


def reinforce(baseline=Baseline.NONE):
    weights = np.zeros(
        (phi_dummy.shape[1], n_actions if USE_GRID_WORLD else action_dim)
    )
    sigma = 2.0  # for Gaussian
    eps = 1.0  # softmax temperature, DO NOT DECAY
    tot_steps = 0
    exp_return_history = np.zeros(max_steps)
    exp_return = expected_return(env_eval, weights, gamma, episodes_eval, eps)
    pbar = tqdm(total=max_steps)

    while tot_steps < max_steps:
        # collect data
        # compute MC return
        # compute gradient of all samples (with/without baseline)
        # average gradient over all samples
        # update weights

        # collect data
        data = collect_data(env, weights, sigma, episodes_per_update)
        phi = np.vstack(data["phi"])
        reward = np.vstack(data["r"])
        actions = np.vstack(data["a"])
        done = np.vstack(data["done"])

        T = reward.shape[0]
        G = np.zeros((T, 1, 1))
        value = 0
        for i in reversed(range(T)):
            if done[i, 0]:
                value = 0
            value = gamma * value + reward[i, 0]
            G[i] = value

        if USE_GRID_WORLD:
            dlog = dlog_softmax_probs(phi, weights, eps, actions)
        else:
            dlog = dlog_gaussian_probs(phi, weights, sigma, actions)

        B = 0
        if baseline == Baseline.AVERAGE:
            B = G.mean()
        elif baseline == Baseline.OPTIMAL:
            B = (G * np.square(dlog)).mean(0) / (np.square(dlog).mean(0))

        delta = G - B
        gradient = dlog * delta
        weights += alpha * gradient.mean(0)

        exp_return_history[tot_steps : tot_steps + T] = exp_return
        tot_steps += T
        exp_return = expected_return(env_eval, weights, gamma, episodes_eval, eps)
        sigma = max(sigma - T / max_steps, 0.1)

        pbar.set_description(f"G: {exp_return:.3f}")
        pbar.update(T)

    pbar.close()
    return exp_return_history


# https://stackoverflow.com/a/63458548/754136
def smooth(arr, span):
    re = np.convolve(arr, np.ones(span * 2 + 1) / (span * 2 + 1), mode="same")
    re[0] = arr[0]
    for i in range(1, span + 1):
        re[i] = np.average(arr[: i + span])
        re[-i] = np.average(arr[-i - span :])
    return re


def error_shade_plot(ax, data, stepsize, smoothing_window=1, **kwargs):
    y = np.nanmean(data, 0)
    x = np.arange(len(y))
    x = [stepsize * step for step in range(len(y))]
    if smoothing_window > 1:
        y = smooth(y, smoothing_window)
    (line,) = ax.plot(x, y, **kwargs)
    error = np.nanstd(data, axis=0)
    if smoothing_window > 1:
        error = smooth(error, smoothing_window)
    error = 1.96 * error / np.sqrt(data.shape[0])
    ax.fill_between(
        x, y - error, y + error, alpha=0.2, linewidth=0.0, color=line.get_color()
    )


if not USE_GRID_WORLD:
    env_id = "Pendulum-v1"
    env = gymnasium.make(env_id)
    env_eval = gymnasium.make(env_id)
    episodes_eval = 100
    # you'll solve the Pendulum when the empirical expected return is higher than -150
    # but it can get even higher, eg -120
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    n_actions = 0
else:
    env_id = "Gym-Gridworlds/Penalty-3x3-v0"
    env = gymnasium.make(env_id, coordinate_observation=True, max_episode_steps=10000)
    env_eval = gymnasium.make(env_id, coordinate_observation=True, max_episode_steps=10)  # 10 steps only for faster eval
    episodes_eval = 1  # max expected return will be 0.941
    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

# automatically set centers and sigmas
n_centers = [7] * state_dim
state_low = env.observation_space.low
state_high = env.observation_space.high
centers = (
    np.array(
        np.meshgrid(
            *[
                np.linspace(
                    state_low[i] - (state_high[i] - state_low[i]) / n_centers[i] * 0.1,
                    state_high[i] + (state_high[i] - state_low[i]) / n_centers[i] * 0.1,
                    n_centers[i],
                )
                for i in range(state_dim)
            ]
        )
    )
    .reshape(state_dim, -1)
    .T
)
sigmas = (state_high - state_low) / np.asarray(
    n_centers
) * 0.75 + 1e-8  # change sigmas for more/less generalization
get_phi = lambda state: rbf_features(
    state.reshape(-1, state_dim), centers, sigmas
)  # reshape because feature functions expect shape (N, S)
phi_dummy = get_phi(env.reset()[0])  # to get the number of features

# hyperparameters
gamma = 0.99
alpha = 0.1
episodes_per_update = 10
max_steps = 100000 if USE_GRID_WORLD else 1000000
baselines = [Baseline.NONE, Baseline.AVERAGE, Baseline.OPTIMAL]
n_seeds = 10
results_exp_ret = np.zeros(
    (
        len(baselines),
        n_seeds,
        max_steps,
    )
)

fig, axs = plt.subplots(1, 1)
axs.set_prop_cycle(color=["red", "green", "blue"])
axs.set_xlabel("Steps")
axs.set_ylabel("Expected Return")

for i, baseline in enumerate(baselines):
    for seed in range(n_seeds):
        np.random.seed(seed)
        exp_return_history = reinforce(baseline)
        results_exp_ret[i, seed] = exp_return_history
        print(baseline.value, seed)

    plot_args = dict(
        stepsize=1,
        smoothing_window=20,
        label=baseline.value,
    )
    error_shade_plot(
        axs,
        results_exp_ret[i],
        **plot_args,
    )
    axs.legend()

plt.show()
