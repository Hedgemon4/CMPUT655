import numpy as np
#
# data = np.load("a6_gridworld.npz")
# s = data["s"]
# a = data["a"]
# r = data["r"]
# s_next = data["s_next"]
# Q = data["Q"]
# V = data["Q"].max(-1)  # value of the greedy policy
# term = data["term"]
# n = s.shape[0]
#
# test = np.where(a == 1)[0]
# items = np.array(test)
# filter_states = s[items, :]
# print(test)

n_centers = 100

state_1_centers = np.linspace(-0.2, 1.2, n_centers)
state_2_centers = np.linspace(-0.2, 1.2, n_centers)
centers = (
    np.array(np.meshgrid(state_1_centers, state_2_centers)).reshape(2, -1).T
)
print(centers.shape)

state_3_centers = np.linspace(-12, 12, 100)[..., None]
print("Hello world")
print(state_3_centers.shape)
