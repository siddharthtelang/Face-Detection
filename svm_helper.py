import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.decomposition import PCA
from helper_functions import *
from sklearn.metrics import accuracy_score
import cvxopt
import cvxopt.solvers


#Form the Kernel
def get_kernel(x, kernel_type, param=0.006):
    """Construct the Kernel for SVM

    Args:
        x (ndarray): Training data points
        kernel_type (str): kernel type - rbf/poly/linear
        param (float, optional): parameter of the kernel. Defaults to 0.006.

    Returns:
        ndarray: the Kernel
    """
    kernel = np.zeros(shape=(x.shape[0], x.shape[0]), dtype=float)
    for i in range(kernel.shape[0]):
        for j in range(kernel.shape[1]):
            if kernel_type == 'rbf':
                kernel[i][j] = math.exp( -1*(np.linalg.norm(x[i] - x[j])**2)/(param*param) )
            elif kernel_type == 'poly':
                kernel[i][j] = math.pow((np.dot(x[i].T, x[j]) +1), param)
            else:
                kernel[i][j] = np.dot(x[i].T, x[j])
    
    while abs(np.linalg.det(kernel)) <= 2:
        kernel = kernel + 0.001*np.identity(kernel.shape[0])
    print(np.linalg.det(kernel))
    
    return kernel

# evaluate the kernel
def evaluate_kernel(x, y, kernel_type='rbf', param=0.006):
    """Evaluates the kernel at a specific test data

    Args:
        x (ndarray): Training data
        y (ndarray): Test point
        kernel_type (str, optional): rbf or poly. Defaults to 'rbf'.
        param (int, optional): scale of the kernel. Defaults to 10.

    Returns:
        [float]: [kernel evaluated at y]
    """
    val = np.zeros(shape=(x.shape[0]), dtype=float)
    for i in range(x.shape[0]):
        if kernel_type == 'rbf':
            val[i] += math.exp( -1*(np.linalg.norm(x[i] - y)**2)/(param*param) )
        elif kernel_type == 'poly':
            val[i] += math.pow((np.dot(x[i].T, y) +1), param)
        else:
            val[i] = np.dot(x[i].T, y)
        
    return val

# solve dual optimization problem
def solve_dual_optimization(kernel, y_train, training_size, threshold=1e-5, C=1):

        P = cvxopt.matrix(np.outer(y_train,y_train)*kernel)
        q = cvxopt.matrix(np.ones(training_size) * -1)
        A = cvxopt.matrix(y_train, (1,training_size))
        b = cvxopt.matrix(0.0)
        # G = cvxopt.matrix(np.diag(np.ones(training_size) * -1))
        G = cvxopt.matrix(np.vstack((np.eye(training_size)*-1,np.eye(training_size))))
        # h = cvxopt.matrix(np.zeros(training_size))
        h = cvxopt.matrix(np.hstack((np.zeros(training_size), np.ones(training_size) * C)))

        solution = cvxopt.solvers.qp(P, q, G, h, A, b)

        # support vectors
        sv = np.ravel(solution['x'])

        # consider all zero below the threshold
        for i in range(sv.shape[0]):
            if sv[i] <= threshold:
                sv[i] = 0
        
        return sv
    
def get_test_train_validation_adaboost(subjects, types, projected, y, training_size, testing_size):
    # init the training and testing data

    validation_size = subjects*types - testing_size 

    y_train = y[:training_size]
    y_validation = y[:validation_size]
    y_test = y[training_size:]

    training_data = projected[:training_size]
    validation_data = projected[:validation_size]
    testing_data = projected[validation_size:]

    print('training_data size = ', training_size)
    print('Validation size = ', validation_size)
    print('testing_data size = ', testing_size)

    return training_data, validation_data, testing_data, y_train, y_validation, y_test    


def get_accuracy(data, iterations, theta, theta0, y, a):
    score = 0
    for i in range(len(data)):
        test = 0
        for k in range(iterations):
            test += a[k] * np.dot(theta[k].T, data[i]) + theta0[k]
        if (test*y[i] > 0):
            score += 1
    return (score*100)/len(data)
