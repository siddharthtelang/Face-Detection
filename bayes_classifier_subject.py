import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import math
from my_pca import *
from my_mda import *
from helper_functions import *
import random
from bayes_classifier import *
import argparse

# classify subjects
def classify_subjects(dataset_file, subjects, types, usePCA, useMDA):

    # load the data
    data, dataset = load_dataset(dataset_file)

    # flatten the dataset
    flattened = flatten_data(dataset, data, subjects, types)

    # Perform PCA if true
    if usePCA:
        projected = doPCA(flattened)
    
    # Perform MDA if true
    elif useMDA:
        projected = perform_MDA(dataset, flattened, subjects, types)

    print('Before dimension reduction shape = ', flattened.shape)
    print('After dimension reduction shape = ', projected.shape)


    # get the training and testing data set
    training_data, testing_data, training_size, testing_size =\
                        get_training_and_testing_data(projected, dataset, subjects, types)

    # get mean and covariance per class
    mu, cov = get_mean_and_covariance_per_class(training_data, projected, dataset)

    accuracy = BayesClassifier_Subject(testing_data, testing_size, dataset, subjects, mu, cov)
    print('Accuracy of Bayes Classifier = ', accuracy)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', '--set',required=False, default='Data/data.mat', type=str)
    parser.add_argument('-subjects', '--sub',required=False, default='200', type=int)
    parser.add_argument('-types', '--type',required=False, default='3', type=int)
    parser.add_argument('-pca', '--pca',required=False, default=False, type=bool)
    parser.add_argument('-mda', '--mda',required=False, default=False, type=bool)
    args = vars(parser.parse_args())
    dataset_file = args['set']
    subjects = args['sub']
    types = args['type']
    usePCA = args['pca']
    useMDA = args['mda']
    classify_subjects(dataset_file, subjects, types, usePCA=usePCA, useMDA=useMDA)
