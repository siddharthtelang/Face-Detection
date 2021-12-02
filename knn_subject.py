import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import math
from my_pca import *
from my_mda import *
from helper_functions import *
import argparse

# K-NN
def find_knn_accuracy(k, training_data, testing_data, subjects, train_per_subject, test_per_subject):
    score = 0
    actual_tests = len(testing_data)
    for i in range(len(testing_data)):
        test_class = int(i/test_per_subject)
        dist = np.zeros(shape=(len(training_data)))
        for j in range(len(training_data)):
            # d =  np.dot(testing_data[i] - mu[j], np.dot(np.linalg.inv(cov[j]), (testing_data[i] - mu[j]).T))

            # use norm to get the distance between the feature points
            d = np.linalg.norm(testing_data[i] - training_data[j])
            dist[j] = d
        
        # sort the distance in ascending order
        sort = np.argsort(dist)
        
        # array to store nearest classes
        predicted_nearest_class = np.zeros(shape=dist.shape[0])
        # votes array
        votes_class = np.zeros(shape=subjects)

        # for every nearest neighbor, check the class and assign vote
        for l in range(k):
            predicted_nearest_class[l] = int(sort[l]/train_per_subject)
            temp_class = int(predicted_nearest_class[l])
            votes_class[temp_class] += 1
        
        # check if any class has same number of votes, if yes then skip the test point
        # we have not designed a tie-breaking algorithm
        same_votes = (np.where(votes_class == np.max(votes_class)))[0]
        if len(same_votes) > 1:
            # print('Same votes, skip this sample')
            actual_tests -= 1
            continue

        # get the class with max votes
        votes_class = -1*votes_class
        predicted_class = np.argsort(votes_class)[0]

        # print(predicted_class)
        # print(test_class)

        # compare with test class
        if predicted_class == test_class:
            score += 1
            # print('Correct, score = ', score)
        # else:
        #     print('Incorrect')
        # print('-------------------------------------------')

    accuracy = score*100/actual_tests
    print('Accuracy of ',str(k),'-NN = ', accuracy)
    return accuracy


def classification_subject(dataset_file, subjects, types, usePCA, useMDA):
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

    # print(projected) 

    print('Before dimension reduction shape = ', flattened.shape)
    print('After dimension reduction shape = ', projected.shape)
    
    training_data, testing_data, train_per_subject, test_per_subject = \
        get_training_and_testing_data_for_knn(projected, subjects, types)
    
    # calculate covariance and mean for each sample
    # The Mannhabolis distance will require this data
    # cov, mu = = calculate_covariance_mean_knn(training_data, dataset)

    accuracy = np.zeros(train_per_subject)
    
    for k in range(train_per_subject):
        accuracy[k] = find_knn_accuracy(k+1, training_data, testing_data,\
                      subjects, train_per_subject, test_per_subject)
    
    nn = np.array([i for i in range(1,train_per_subject+1)])
    plt.title('K-NN Classification ' + dataset)
    plt.xlabel('Number of nearesr neighbors')
    plt.ylabel('Accuracy')
    # plt.ylim(50,100)
    plt.plot(nn, accuracy)
    plt.show()


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
    classification_subject(dataset_file, subjects, types, usePCA, useMDA)