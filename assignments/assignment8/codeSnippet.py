import numpy as np

def dlog_gaussian_probs(phi: np.array,  weights: np.array,  sigma: np.array, action: np.array):
    # implement log-derivative of pi with respect to the mean only
    # diag_covar_inverse = np.linalg.inv(np.square(np.diag(np.full(action.shape[0], sigma))))
    diag_covar_inverse = np.diag(np.full(action.shape[0], (1 / sigma) ** 2))
    return diag_covar_inverse * (action - np.dot(phi, weights)) * phi

n_actions = 5

def softmax_probs(phi, weights, eps):
    q = np.dot(phi, weights)
    # this is a trick to make it more stable
    # see https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
    q_exp = np.exp((q - np.max(q, -1, keepdims=True)) / max(eps, 1e-12))
    probs = q_exp / q_exp.sum(-1, keepdims=True)
    return probs

def dlog_softmax_probs(phi, weights, eps, act):
    # implement log-derivative of pi
    phi_sa = phi[..., None].repeat(n_actions, axis=-1)
    # Make mask
    mask = np.zeros((act.shape[0], phi.shape[1], n_actions))
    for i, num in enumerate(act[:, 0]):
        mask[i][:, num] = 1
    probs = softmax_probs(phi, weights, eps)
    return phi_sa * mask - phi_sa * probs[:, None]


print(dlog_softmax_probs(np.array([[[1, 0, 0, 0, 0]]]), np.array([[0, 0, 0, 0, 0]]), 1.0, np.array([0])))
