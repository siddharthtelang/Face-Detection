import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.decomposition import PCA
from helper_functions import *
from bayes_classifier import *
from my_lda import *
import argparse


def classify_pose_Bayes(dataset_file, subjects, types, usePCA, useMDA, training_size):

    # load the data
    data, dataset = load_dataset(dataset_file)

    # flatten the dataset
    flattened, y = flatten_data_for_pose_classification(data, subjects, types)

    # Perform PCA if true
    if usePCA:
        projected = doPCA(flattened)
    
    # Perform MDA if true
    elif useMDA:
        projected = perform_LDA(flattened, subjects, types)

    print('Before dimension reduction shape = ', flattened.shape)
    print('After dimension reduction shape = ', projected.shape)

    training_data, training_size, testing_data, testing_size, y_train, y_test = \
        get_training_testing_data_for_pose_classification\
            (projected, y, subjects, types, training_size)

    # build up the expression and neutral training set
    expression_size = int(training_size / 2)
    neutral_size = expression_size
    expression = np.zeros(shape=(expression_size, projected.shape[1]))
    neutral = np.zeros(shape=(neutral_size, projected.shape[1]))
    c = 0
    for i in range(0, training_size, 2):
        neutral[c] = projected[i]
        expression[c] = projected[i+1]
        c += 1

    """### Calculate class mean and covariance"""

    # mean
    mu_expression = np.mean(expression, axis=0)
    mu_neutral = np.mean(neutral, axis=0)

    # covariance expression
    mat = expression - mu_expression
    cov_expression = (np.dot(mat.T, mat)) / expression_size

    #covariance neutral
    mat = neutral - mu_neutral
    cov_neutral = (np.dot(mat.T, mat)) / neutral_size

    # check for zero determinant, add noise if zero
    while abs(np.linalg.det(cov_expression)) <= 2:
        cov_expression = cov_expression + 0.001*np.identity(cov_expression.shape[0])
    print(np.linalg.det(cov_expression))

    while abs(np.linalg.det(cov_neutral)) <= 2:
        cov_neutral = cov_neutral + 0.001*np.identity(cov_neutral.shape[0])
    print(np.linalg.det(cov_neutral))

    """# Test data using ML"""
    accuracy = BayesClassifier_Pose(testing_data, testing_size, y_test, cov_neutral, \
                                    mu_neutral, cov_expression, mu_expression)
    print('Accuracy of ML Estimate = ',accuracy)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', '--set',required=False, default='Data/data.mat', type=str)
    parser.add_argument('-subjects', '--sub',required=False, default='200', type=int)
    parser.add_argument('-types', '--type',required=False, default='2', type=int)
    parser.add_argument('-trainingSize', '--trsize',required=False, default='', type=int)
    parser.add_argument('-pca', '--pca',required=False, default=False, type=bool)
    parser.add_argument('-mda', '--mda',required=False, default=False, type=bool)
    args = vars(parser.parse_args())
    dataset_file = args['set']
    subjects = args['sub']
    types = args['type']
    usePCA = args['pca']
    useMDA = args['mda']
    training_size = args['trsize']
    classify_pose_Bayes(dataset_file, subjects, types, usePCA, useMDA, training_size=training_size)
