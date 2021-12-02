import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.decomposition import PCA
from helper_functions import *
from my_lda import *
import argparse

def knn_classification(nn, acc, training_data, testing_data, y_train, y_test):

    for k in nn:
        score = 0
        actual_tests = len(testing_data)

        for i in range(1,len(testing_data)):
            neutral_score = 0
            expression_score = 0

            test_class = y_test[i]
            dist = np.zeros(shape=(len(training_data)))
            
            for j in range(len(training_data)):
                # d =  np.dot(testing_data[i] - mu[j], np.dot(np.linalg.inv(cov[j]), (testing_data[i] - mu[j]).T))
                # Calculate the Norm
                d = np.linalg.norm(testing_data[i] - training_data[j])
                dist[j] = d
            
            # sort the distances
            sort = np.argsort(dist)
            
            predicted_nearest_class = np.zeros(shape=dist.shape[0])

            # check the class of each NN and count the votes per class
            for l in range(k):
                predicted_nearest_class[l] = y_train[int(sort[l])]
                if predicted_nearest_class[l] == 1:
                    neutral_score += 1
                else:
                    expression_score += 1

            if neutral_score > expression_score:
                predicted_class = 1
            elif neutral_score < expression_score:
                predicted_class = -1
            elif neutral_score == expression_score:
                # print('Tie')
                actual_tests -= 1
                continue

            if predicted_class == test_class:
                score += 1
    
        accuracy = (score*100/actual_tests)
        print('Accuracy of ',str(k),'-NN = ', accuracy)
        acc.append(accuracy)


def classify_pose_KNN(dataset_file, subjects, types, usePCA, useMDA, training_size):

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
    
    # Evaluate for 10 Nearest Neighbors
    nn = [i for i in range(1,11)]
    acc = []
    knn_classification(nn, acc, training_data, testing_data, y_train, y_test)

    #plot the results
    plt.figure(figsize=(10,10))
    plt.ylim(50,100)
    plt.xlabel('Number of NN')
    plt.ylabel('Accuracy')
    plt.plot(nn, acc)
    plt.show()
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', '--set',required=False, default='Data/data.mat', type=str)
    parser.add_argument('-subjects', '--sub',required=False, default='200', type=int)
    parser.add_argument('-types', '--type',required=False, default='2', type=int)
    parser.add_argument('-trainingSize', '--trsize',required=False, default='200', type=int)
    parser.add_argument('-pca', '--pca',required=False, default=False, type=bool)
    parser.add_argument('-mda', '--mda',required=False, default=False, type=bool)
    args = vars(parser.parse_args())
    dataset_file = args['set']
    subjects = args['sub']
    types = args['type']
    usePCA = args['pca']
    useMDA = args['mda']
    training_size = args['trsize']
    classify_pose_KNN(dataset_file, subjects, types, usePCA, useMDA, training_size)