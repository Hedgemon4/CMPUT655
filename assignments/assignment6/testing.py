import numpy as np

#
# def rbf_features(
#     state: np.array,  # (N, S)
#     centers: np.array,  # (D, S)
#     sigmas: float,
# ) -> np.array:  # (N, D)
#     """
#     Computes exp(- ||state - centers||**2 / sigmas**2 / 2).
#     """
#     Z = np.zeros((state.shape[0], centers.shape[0]))
#     for i in range(state.shape[0]):
#         for j in range(centers.shape[0]):
#             Z[i, j] = np.linalg.norm(state[i, :] - centers[j, :])
#     return np.exp((Z ** 2) * -1 / (sigmas**2) / 2)
#
#
# def tile_features(
#         state: np.array,  # (N, S)
#         centers: np.array,  # (D, S)
#         widths: float,
#         offsets: list = [0],  # list of tuples of length S
# ) -> np.array:  # (N, D)
#     """
#     Given centers and widths, you first have to get an array of 0/1, with 1s
#     corresponding to tile the state belongs to.
#     If "offsets" is passed, it means we are using multiple tilings, i.e., we
#     shift the centers according to the offsets and repeat the computation of
#     the 0/1 array. The final output will sum the "activations" of all tilings.
#     We recommend to normalize the output in [0, 1] by dividing by the number of
#     tilings (offsets).
#     Recall that tiles are squares, so you can't use the L2 Euclidean distance to
#     check if a state belongs to a tile, but the L1 distance.
#     Note that tile coding is more general and allows for rectangles (not just squares)
#     but let's consider only squares for the sake of simplicity.
#     """
#     Z = np.zeros((state.shape[0], centers.shape[0]))
#     for i in range(state.shape[0]):
#         for j in range(centers.shape[0]):
#             for offset in offsets:
#                 Z[i, j] += 1 if np.all(np.abs(centers[:, i] - state[j, :]) < (widths + offset)) else 0
#     return Z / len(offsets)
#
#
X = np.arange(0, 8).reshape(2, 4)
Y = np.arange(0, 12).reshape(4, 3)
print(X.shape[0])
print(X)
print(Y)
#
# print(tile_features(X, Y, 0.1))
# width = 10
# x = (1, 2, 3)
# print(np.asarray(x) + width)
print(np.argmax(X, axis=1, keepdims=True) + np.zeros(X.shape))
print(X.argmax(axis=1))
