import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.decomposition import PCA
import random

def perform_LDA(flattened, subjects, types):
    dataset = 'face'

    noise = {'pose':0.93, 'face':0.999}

    neutral = np.zeros(shape=(subjects, flattened.shape[1]))
    expression = np.zeros(shape=(subjects, flattened.shape[1]))

    c = 0
    for i in range(0, subjects*types, 2):
        neutral[c] = flattened[i]
        expression[c] = flattened[i+1]
        c+=1
    mu_neutral = np.mean(neutral, axis=0)
    mu_expression = np.mean(expression, axis=0)

    mat = neutral - mu_neutral
    cov_neutral = np.dot(mat.T, mat) / subjects
    mat = expression - mu_neutral
    cov_expression = np.dot(mat.T, mat) / subjects
    mat = (mu_neutral - mu_expression).reshape((1, flattened.shape[1]))
    # Between class scatter
    sigma_b = np.dot(mat.T, mat)
    # Within class scatter
    sigma_w = cov_neutral + cov_neutral

    # while abs(np.linalg.det(sigma_b)) <= 2:
    sigma_b = sigma_b + 0.999*np.identity(sigma_b.shape[0])
    print(np.linalg.det(sigma_b))

    # while abs(np.linalg.det(sigma_w)) <= 2:
    sigma_w = sigma_w + 0.868*np.identity(sigma_w.shape[0])
    print(np.linalg.det(sigma_w))

    # Find the optimal direction
    a = np.dot(np.linalg.inv(sigma_w), sigma_b)
    val, vec = np.linalg.eig(a)

    # sort the eigen values and vectors in descending order
    idx = val.argsort()[::-1]
    val_ = val[idx]
    vec_  = vec[:, idx]

    dim = types - 1

    final_vector = np.dot(np.linalg.inv(sigma_w), (mu_neutral - mu_expression))
    final_vector = final_vector.reshape((final_vector.shape[0],1))

    f = np.dot(flattened, final_vector)

    return f


