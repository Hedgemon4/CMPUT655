import numpy as np
import gymnasium
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.preprocessing import PolynomialFeatures  # makes poly features super easy

np.set_printoptions(precision=3, suppress=True)

# Notation for array sizes:
# - S: state dimensionality
# - D: features dimensionality
# - N: number of samples
#
# N is always the first dimension, meaning that states come in arrays of shape
# (N, S) and features in arrays of shape (N, D).
# We recommend to implement the functions below assuming that the input has
# always shape (N, S) and the output (N, D), even when N = 1.


def poly_features(state: np.array, degree: int) -> np.array:
    """
    Compute polynomial features. For example, if state = (s1, s2) and degree = 2,
    the output must be [1, s1, s2, s1*s2, s1**2, s2**2].
    """
    return PolynomialFeatures(degree).fit_transform(state)


def rbf_features(
    state: np.array,  # (N, S)
    centers: np.array,  # (D, S)
    sigmas: float,
) -> np.array:  # (N, D)
    """
    Computes exp(- ||state - centers||**2 / sigmas**2 / 2).
    """
    Z = np.zeros((state.shape[0], centers.shape[0]))
    for i in range(state.shape[0]):
        for j in range(centers.shape[0]):
            Z[i, j] = np.linalg.norm(state[i, :] - centers[j, :])
    return np.exp((Z**2) * -1 / (sigmas**2) / 2)


def tile_features(
    state: np.array,  # (N, S)
    centers: np.array,  # (D, S)
    widths: float,
    offsets: list = [0],  # list of tuples of length S
) -> np.array:  # (N, D)
    """
    Given centers and widths, you first have to get an array of 0/1, with 1s
    corresponding to tile the state belongs to.
    If "offsets" is passed, it means we are using multiple tilings, i.e., we
    shift the centers according to the offsets and repeat the computation of
    the 0/1 array. The final output will sum the "activations" of all tilings.
    We recommend to normalize the output in [0, 1] by dividing by the number of
    tilings (offsets).
    Recall that tiles are squares, so you can't use the L2 Euclidean distance to
    check if a state belongs to a tile, but the L1 distance.
    Note that tile coding is more general and allows for rectangles (not just squares)
    but let's consider only squares for the sake of simplicity.
    """
    Z = np.zeros((state.shape[0], centers.shape[0]))
    for i in range(state.shape[0]):
        for j in range(centers.shape[0]):
            for offset in offsets:
                Z[i, j] += (
                    1
                    if np.all(
                        np.abs(state[i, :] - centers[j, :])
                        < (widths + np.asarray(offset))
                    )
                    else 0
                )
    return Z / len(offsets)


def coarse_features(
    state: np.array,  # (N, S)
    centers: np.array,  # (D, S)
    widths: float,
    offsets: list = [0],  # list of tuples of length S
) -> np.array:  # (N, D)
    """
    Same as tile coding, but we use circles instead of squares, so use the L2
    Euclidean distance to check if a state belongs to a circle.
    Note that coarse coding is more general and allows for ellipses (not just circles)
    but let's consider only circles for the sake of simplicity.
    """
    Z = np.zeros((state.shape[0], centers.shape[0]))
    for i in range(state.shape[0]):
        for j in range(centers.shape[0]):
            for offset in offsets:
                Z[i, j] += (
                    1
                    if np.all(
                        np.linalg.norm(state[i, :] - centers[j, :])
                        < (widths + np.asarray(offset))
                    )
                    else 0
                )
    return Z / len(offsets)


def aggregation_features(state: np.array, centers: np.array) -> np.array:
    """
    Aggregate states to the closest center. The output will be an array of 0s and
    one 1 corresponding to the closest tile the state belongs to.
    Note that we can turn this into a discrete (finite) representation of the state,
    because we will have as many feature representations as centers.
    """
    Z = np.zeros((state.shape[0], centers.shape[0]))
    for i in range(state.shape[0]):
        for j in range(centers.shape[0]):
            Z[i, j] = np.linalg.norm(state[i, :] - centers[j, :])
    output = np.zeros_like(Z)
    output[np.arange(state.shape[0]), np.argmin(Z, axis=1)] = 1.0
    return output


state_size = 2
n_samples = 10
n_centers = 100
state = np.random.rand(n_samples, state_size)  # in [0, 1]

state_1_centers = np.linspace(-0.2, 1.2, n_centers)
state_2_centers = np.linspace(-0.2, 1.2, n_centers)
centers = (
    np.array(np.meshgrid(state_1_centers, state_2_centers)).reshape(state_size, -1).T
)  # makes a grid of uniformly spaced centers in the plane [-0.2, 1.2]^2
sigmas = 0.2
widths = 0.2
offsets = [(-0.1, 0.0), (0.0, 0.1), (0.1, 0.0), (0.0, -0.1)]

poly = poly_features(state, 2)
aggr = aggregation_features(state, centers)
rbf = rbf_features(state, centers, sigmas)
tile_one = tile_features(state, centers, widths)
tile_multi = tile_features(state, centers, widths, offsets)
coarse_one = coarse_features(state, centers, widths)
coarse_multi = coarse_features(state, centers, widths, offsets)

fig, axs = plt.subplots(1, 6)
extent = [
    state_1_centers[0],
    state_1_centers[-1],
    state_2_centers[0],
    state_2_centers[-1],
]  # to change imshow axes
axs[0].imshow(rbf[0].reshape(n_centers, n_centers), extent=extent, origin="lower")
axs[1].imshow(tile_one[0].reshape(n_centers, n_centers), extent=extent, origin="lower")
axs[2].imshow(
    tile_multi[0].reshape(n_centers, n_centers), extent=extent, origin="lower"
)
axs[3].imshow(
    coarse_one[0].reshape(n_centers, n_centers), extent=extent, origin="lower"
)
axs[4].imshow(
    coarse_multi[0].reshape(n_centers, n_centers), extent=extent, origin="lower"
)
axs[5].imshow(aggr[0].reshape(n_centers, n_centers), extent=extent, origin="lower")
titles = [
    "RBFs",
    "Tile (1 Tiling)",
    "Tile (4 Tilings)",
    "Coarse (1 Field)",
    "Coarse (4 Fields)",
    "Aggreg.",
]  # we can't plot poly like this
for ax, title in zip(axs, titles):
    ax.plot(state[0][0], state[0][1], marker="+", markersize=12, color="red")
    ax.set_title(title)
plt.suptitle(f"State {state[0]}")
plt.show()

#################### PART 1
# Submit your heatmaps.
# What are the hyperparameters of each FA and how do they affect the shape of
# the function they can approximate?
# - In RBFs the hyperparameter(s) is/are ... More/less ... will affect ...,
#   while narrower/wider ... will affect ...
# - In tile/coarse coding the hyperparameter(s) is/are ...
# - In polynomials the hyperparameter(s) is/are ...
# - In state aggregation the hyperparameter(s) is/are ...
# Discuss each bullet point in at most two sentences.

#################### PART 2
# Consider the function below.

x = np.linspace(-10, 10, 100)
y = np.sin(x) + x**2 - 0.5 * x**3 + np.log(np.abs(x))
fig, axs = plt.subplots(1, 1)
axs.plot(x, y)
plt.show()

# With SL, (try to) train a linear approximation to fit the above function (y)
# using gradient descent.
# Start with all weights to 0.
# Feel free to use a better learning rate (maybe check the book for suggestions)
# and more iterations.
#
# - Use all 5 FAs implemented above and submit your plots. Select the
#   hyperparameters (degree, centers, sigmas, width, offsets) to achieve the best
#   results (lowest MSE).
# - Assume the 2nd dimension of the state is in the range [0, 1000] instead of [-10, 10].
#   Also, assume that your implementation of RBFs, Tile Coding, and Coarse Coding
#   allows to pass different widths/sigmas for every dimension of the state.
#   How would you change the hyperparameters? Would you have more centers?
#   Or wider sigmas/widths? Both? Justify your answer in one sentence.
#
# Note: we don't want you to achieve MSE 0. Just have a decent plot with each FA,
# or discuss if some FA is not suitable to fit the given function, and report your plots.
# Anything like the demo plot is fine.


max_iter = 10000
thresh = 1e-8
alpha = 0.01
number_of_centers = 30
rbf_centers = np.linspace(-12, 12, 500)[..., None]
tile_centers = np.linspace(-12, 12, 500)[..., None]
coarse_centers = np.linspace(-12, 12, 500)[..., None]
aggregation_centers = np.linspace(-12, 12, 500)[..., None]
offsets = [(-1.0, 0.0), (0.0, 1.0), (1.0, 0.0), (0.0, -1.0)]

for name, get_phi in zip(
    ["Poly", "RBFs", "Tiles", "Coarse", "Aggreg."],
    [
        lambda state: poly_features(state, 3),
        lambda state: rbf_features(state, rbf_centers, 0.6),
        lambda state: tile_features(state, tile_centers, 1, offsets),
        lambda state: coarse_features(state, coarse_centers, 1, offsets),
        lambda state: aggregation_features(state, aggregation_centers),
    ],
):
    phi = get_phi(x[..., None])  # from (N,) to (N, S) with S = 1
    weights = np.zeros(phi.shape[-1])
    pbar = tqdm(total=max_iter)
    # normalized_phi = (phi - phi.min()) / (phi.max() - phi.min())

    # l1_norm_row = np.sum(np.abs(phi), axis=1, keepdims=True)
    # l1_norm_row[l1_norm_row == 0] = 1
    # normalized_phi = phi / l1_norm_row
    
    for iter in range(max_iter):
        # do gradient descent
        test_mse = (y - np.dot(phi, weights)) ** 2
        mse = np.mean(test_mse)
        pbar.set_description(f"MSE: {mse}")
        pbar.update()
        norm_phi = np.linalg.norm(phi)
        if norm_phi == 0:
            norm_phi = 1
        weights += np.mean(alpha * (y - np.dot(phi, weights))[..., None] * phi / norm_phi, axis=0)
        if mse < thresh:
            break

    print(f"Iterations: {iter}, MSE: {mse}, N. of Features {len(weights)}")
    fig, axs = plt.subplots(1, 2)
    axs[0].plot(x, y)
    y_hat = np.dot(phi, weights)
    axs[1].plot(x, y_hat)
    axs[0].set_title("True Function")
    axs[1].set_title(f"Approximation with {name} (MSE {mse:.3f})")
    plt.show()

# Now repeat the experiment but fit the following function y.
# Submit your plots and discuss your results, paying attention to the
# non-smoothness of the new target function.
# - How did you change your hyperparameters? Did you use more/less wider/narrower features?
# - Consider the number of features. How would it change if your state would be 2-dimensional?
# Discuss each bullet point in at most two sentences.

x = np.linspace(-10, 10, 100)
y = np.zeros(x.shape)
y[0:10] = x[0:10] ** 3 / 3.0
y[10:20] = np.exp(x[25:35])
y[20:30] = -x[0:10] ** 3 / 2.0
y[30:60] = 100.0
y[60:70] = 0.0
y[70:100] = np.cos(x[70:100]) * 100.0
fig, axs = plt.subplots(1, 1)
axs.plot(x, y)
plt.show()

number_of_centers = 500
centers = np.linspace(-10, 10, number_of_centers)[..., None]
offsets = [(-1.0, 0.0), (0.0, 1.0), (1.0, 0.0), (0.0, -1.0)]

# widths = 0.5
widths = 0.5

for name, get_phi in zip(
    ["Poly", "RBFs", "Tiles", "Coarse", "Aggreg."],
    [
        lambda state: poly_features(state, 3),
        lambda state: rbf_features(state, centers, 0.2),
        lambda state: tile_features(state, centers, widths),
        lambda state: coarse_features(state, centers, widths),
        lambda state: aggregation_features(state, centers),
    ],
):
    phi = get_phi(x[..., None])
    weights = np.zeros(phi.shape[-1])
    pbar = tqdm(total=max_iter)
    for iter in range(max_iter):
        # do gradient descent
        test_mse = (y - np.dot(phi, weights)) ** 2
        mse = np.mean(test_mse)
        pbar.set_description(f"MSE: {mse}")
        pbar.update()
        norm_phi = np.linalg.norm(phi)
        if norm_phi == 0:
            norm_phi = 1
        weights += np.mean(alpha * (y - np.dot(phi, weights))[..., None] * phi / norm_phi, axis=0)
        if mse < thresh:
            break

    print(f"Iterations: {iter}, MSE: {mse}, N. of Features {len(weights)}")
    fig, axs = plt.subplots(1, 2)
    axs[0].plot(x, y)
    y_hat = np.dot(phi, weights)
    axs[1].plot(x, y_hat)
    axs[0].set_title("True Function")
    axs[1].set_title(f"Approximation with {name} (MSE {mse:.3f})")
    plt.show()

#################### PART 3
# Consider the Gridworld depicted below. The dataset below contains episodes
# collected using the optimal policy, and the heatmap below shows its V-function.
# - Consider the 5 FAs implemented above and discuss why each would be a
#   good/bad choice. Discuss each in at most two sentences.

# The given data is a dictionary of (s, a, r, s', term, Q) arrays.
# Unlike the previous assignments, the state is the (x, y) coordinate of the agent
# on the grid.
# - Run batch semi-gradient TD prediction with a FA of your choice (the one you
#   think would work best) to learn an approximation of the V-function.
#   Use gamma = 0.99. Increase the number of iterations, if you'd like.
#   Plot your result of the true V-function against your approximation using the
#   provided plotting function.

data = np.load("a6_gridworld.npz")
s = data["s"]
a = data["a"]
r = data["r"]
s_next = data["s_next"]
Q = data["Q"]
V = data["Q"].max(-1)  # value of the greedy policy
term = data["term"]
n = s.shape[0]
gamma = 0.99

# needed for heatmaps
s_idx = np.ravel_multi_index(s.T, (9, 9))
unique_s_idx = np.unique(s_idx, return_index=True)[1]

fig, axs = plt.subplots(1, 1)
# surf = axs.tricontourf(s[:, 0], s[:, 1], V)
surf = axs.imshow(V[unique_s_idx].reshape(9, 9))
plt.colorbar(surf)
plt.show()

### Old Params
# max_iter = 20000
# alpha = 0.05
# thresh = 1e-8
#
# number_of_centers = 100
# state_1_centers = np.linspace(0, 9, number_of_centers)
# state_2_centers = np.linspace(0, 9, number_of_centers)
# centers = (
#     np.array(np.meshgrid(state_1_centers, state_2_centers)).reshape(state_size, -1).T
# )
# offsets = [(-1, 3), (1, 3), (-1, -3), (1, 3), (2, 6), (-2, 6), (2, -6), (-2, -6), (3, 9), (-3, 9), (3, -9), (-3, -9)]


max_iter = 60000
# max_iter = 20000
alpha = 0.1
thresh = 1e-8

number_of_centers = 20
state_1_centers = np.linspace(0, 8, number_of_centers)
state_2_centers = np.linspace(0, 8, number_of_centers)
centers = (
    np.array(np.meshgrid(state_1_centers, state_2_centers)).reshape(state_size, -1).T
)

name, get_phi = "RBFs", lambda state : rbf_features(state, centers, 0.3)

# number_of_centers = 10
# state_1_centers = np.linspace(0, 8, number_of_centers)
# state_2_centers = np.linspace(0, 8, number_of_centers)
# centers = (
#     np.array(np.meshgrid(state_1_centers, state_2_centers)).reshape(state_size, -1).T
# )
# offsets = [(-.1, 0), (0, -.1), (.1, 0), (0, .1), (-0.05, 0), (0, -0.05), (0.05, 0), (0, 0.05)]
# name, get_phi = "Tiles", lambda state: tile_features(state, centers, 1, offsets)

# should try a non-uniform amount of offset

# Pick one
# name, get_phi = "Poly", lambda state : poly_features(state, ...)
# name, get_phi = "RBFs", lambda state : rbf_features(state, centers, 0.3)
# name, get_phi = "Tiles", lambda state: tile_features(state, centers, 5, offsets)
# name, get_phi = "Coarse", lambda state : coarse_features(state, ...)
# name, get_phi = "Aggreg.", lambda state: aggregation_features(state, ...)

# state, centers, widths, offsets
term_nums = term.astype(int)

phi = get_phi(s)
phi_next = get_phi(s_next)
normalized_phi = (phi - phi.min()) / (phi.max() - phi.min())
weights = np.zeros(phi.shape[-1])
pbar = tqdm(total=max_iter)
for iter in range(max_iter):
    # do TD semi-gradient
    td_error = r + (gamma * np.dot(phi_next, weights) * (1 - term_nums)) - np.dot(phi, weights)
    weights += np.mean(alpha * (td_error)[..., None] * phi, axis=0)
    # mse = ...  # prediction - V
    mse = np.mean((np.dot(phi, weights) - V) ** 2)
    pbar.set_description(f"TDE: {td_error}, MSE: {mse}")
    pbar.update()
    if mse < thresh:
        break

td_prediction = np.dot(phi, weights)

print(f"Iterations: {iter}, MSE: {mse}, N. of Features {len(weights)}")
fig, axs = plt.subplots(1, 2)
axs[0].imshow(V[unique_s_idx].reshape(9, 9))
axs[1].imshow(td_prediction[unique_s_idx].reshape(9, 9))
axs[0].set_title("V")
axs[1].set_title(f"Approx. with RBFs (MSE {mse:.3f})")
plt.show()


#################### PART 4
# - Run TD again, but this time learn an approximation of the Q-function.
#   How did you have to change your code?

# TODO: Loop over the actions because we can't update it all at once

# I made an attempt :(
# number_of_centers = 16
# state_1_centers = np.linspace(0, 8, number_of_centers)
# state_2_centers = np.linspace(0, 8, number_of_centers)
# centers = (
#     np.array(np.meshgrid(state_1_centers, state_2_centers)).reshape(state_size, -1).T
# )
# offsets = [(-.1, 0), (0, -.1), (.1, 0), (0, .1), (-0.05, 0), (0, -0.05), (0.05, 0), (0, 0.05)]
# name, get_phi = "Tiles", lambda state: tile_features(state, centers, 4, offsets)

max_iter = 1000
alpha = 0.01
thresh = 1e-8
n_actions = 5

phi = get_phi(s)
phi_next = get_phi(s_next)
weights = np.zeros((phi.shape[-1], n_actions))  # you have to change something here
pbar = tqdm(total=max_iter)
for iter in range(max_iter):
    # do TD semi-gradient
    # td_error = ...
    # mse = ...  # prediction - Q
    # do TD semi-gradient
    for action in range(n_actions):
        action_indices = np.array(np.where(a == action)[0])
        phi_actions = phi[action_indices, :]
        phi_next_actions = phi_next[action_indices, :]
        term_nums_actions = term_nums[action_indices]
        td_error = r[action_indices] + (gamma * np.dot(phi_next_actions, weights[:, action]) * (1 - term_nums_actions)) - np.dot(phi_actions, weights[:, action])
        weights[:, action] += np.mean(alpha * (td_error)[..., None] * phi_actions, axis=0)
    # mse = ...  # prediction - V
    mse = np.mean((np.dot(phi, weights) - Q) ** 2)
    pbar.set_description(f"TDE: {td_error}, MSE: {mse}")
    pbar.update()
    if mse < thresh:
        break

print(Q[unique_s_idx].argmax(-1).reshape(9, 9))  # check optimal policy
td_prediction = np.dot(phi, weights)

print(f"Iterations: {iter}, MSE: {mse}, N. of Features {len(weights)}")
fig, axs = plt.subplots(2, n_actions, figsize=(15, 8))
for i, j in zip(range(n_actions), ["LEFT", "DOWN", "RIGHT", "UP", "STAY"]):
    axs[0][i].imshow(Q[unique_s_idx, i].reshape(9, 9))
    axs[1][i].imshow(td_prediction[unique_s_idx, i].reshape(9, 9))
    axs[0][i].set_title(f"Q {j}")
    axs[1][i].set_title(f"Approx. with RBFs (MSE {mse:.3f})")
plt.subplots_adjust(left=0.05, right=0.95, top=0.85, bottom=0.1, hspace=0.6, wspace=0.4)
plt.show()


#################### PART 5
# Discuss similarities and differences between SL regression and RL TD regression.
# - Discuss loss functions, techniques applicable to minimize it, and additional
#   challenges of RL.
# - What are the differences between "gradient descent" and "semi-gradient
#   descent" for TD?
# - Assume you'd have to learn the Q-function when actions are continuous.
#   How would you change your code?
