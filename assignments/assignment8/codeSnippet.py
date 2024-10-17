import numpy as np

def dlog_gaussian_probs(phi: np.array,  weights: np.array,  sigma: np.array, action: np.array):
    # implement log-derivative of pi with respect to the mean only
    # diag_covar_inverse = np.linalg.inv(np.square(np.diag(np.full(action.shape[0], sigma))))
    diag_covar_inverse = np.diag(np.full(action.shape[0], (1 / sigma) ** 2))
    return diag_covar_inverse * (action - np.dot(phi, weights)) * phi
