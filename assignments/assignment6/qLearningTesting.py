import numpy as np

data = np.load("a6_gridworld.npz")
s = data["s"]
a = data["a"]
r = data["r"]
s_next = data["s_next"]
Q = data["Q"]
V = data["Q"].max(-1)  # value of the greedy policy
term = data["term"]
n = s.shape[0]

test = np.where(a == 1)[0]
items = np.array(test)
filter_states = s[items, :]
print(test)
