import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import math
from my_pca import *
import random

# function to load the data set from mat file
def load_dataset(dataset_file):

    data = sio.loadmat(dataset_file)

    if 'pose' in dataset_file:
        data = data.get('pose')
        dataset = 'pose'
    elif 'illumination' in dataset_file:
        data = data.get('illum')
        dataset = 'illum'
    else:
        data = data.get('face')
        dataset = 'face'
    
    return data, dataset

# flattens the data
def flatten_data(dataset, data, subjects, types):
    if dataset == 'pose':
        flattened = np.zeros(shape=(subjects*types, data.shape[0]*data.shape[1]))
    elif dataset == 'illum':
        flattened = np.zeros(shape=(subjects*types, data.shape[0]))
    else:
        flattened = np.zeros(shape=(subjects*types, data.shape[0]*data.shape[1]))

    c = 0
    d = 0
    for i in range(flattened.shape[0]):
        if c == types:
            c = 0
            d += 1
        if dataset == 'pose':
            temp = data[:,:,c,d]
            flattened[i] = temp.flatten()
        elif dataset == 'face':
            temp = data[:,:,i]
            flattened[i] = temp.flatten()
        elif dataset == 'illum':
            flattened[i] = data[:,c,d]
        c += 1
    return flattened

# flattens data for pose classification
def flatten_data_for_pose_classification(data, subjects, types):
    # flatten the dataset
    flattened = np.zeros(shape=(subjects*types, data.shape[0]*data.shape[1]))

    # label for neutral and expression - 0 / 1
    y = np.ones(shape=(subjects*types)) # by default all are one
    c = 0
    for i in range(0, data.shape[2], 3):
        temp1 = data[:,:,i]
        temp2 = data[:,:,i+1]
        flattened[c] = temp1.flatten()
        flattened[c+1] = temp2.flatten()
        y[c+1] = -1 # expression label -1
        c += 2

    return flattened, y

# builds up training and testing data
def get_training_and_testing_data(projected, dataset, subjects, types, reduced=True):
    
    training_data = []
    testing_data = []

    if dataset == 'face':
        c = 1 # counter to pick random sample from either illumination or pose
        for i in range(0, projected.shape[0], 3):
            training_data.append(projected[i+2])
            # if c % 2 == 0:
            training_data.append(projected[i+1])
            testing_data.append(projected[i])
            # else:
            #     training_data.append(projected[i])
            #     testing_data.append(projected[i+2])
            # c += 1
        testing_size = len(testing_data)
        training_size = len(training_data)
    else:
        if reduced:
            size = projected.shape[1]
        else:
            size = projected.shape[0]*projected.shape[1]
        training_size = math.ceil(types*(2/3))
        testing_size = types - training_size
        subject = 0 # counter

        for i in range(subjects):
            temp = dict()
            temp['data'] = []

            for j in range(types):
                if reduced:
                    temp['data'].append((projected[subject]))
                    subject += 1
                else:
                    temp['data'].append((projected[:,:,j,i]).flatten())

            random.shuffle(temp['data']) # shuffle the data
            tr = temp['data'][:training_size] # training batch
            te = temp['data'][training_size:] # testing batch
            # te = temp['data'][:testing_size]
            # tr = temp['data'][testing_size:]

            training_data.append({'class': i, 'data':tr})
            testing_data.append({'class': i, 'data': te})
    
    return training_data, testing_data, training_size, testing_size


# Split in training and testing data set for KNN
def get_training_and_testing_data_for_knn(projected, subjects, types):
    training_data = []
    testing_data = []
    train_per_subject = int(math.ceil(2*types/3))
    test_per_subject = types - train_per_subject
    print('Training data per subject=', train_per_subject)
    print('Testing data per subject=', test_per_subject)

    for i in range(subjects):
        temp = []
        start = i*types
        end = (i+1)*types
        
        for j in range(start , start + test_per_subject):
            testing_data.append(projected[j])
        
        for j in range(start + test_per_subject , end):
            training_data.append(projected[j])
        
        # training_data.append(projected[start : start + train_per_subject])
        # testing_data.append(projected[start + train_per_subject : end])
    print('Size of training data = ', len(training_data))
    print('Size of testing data = ', len(testing_data))
    return training_data, testing_data, train_per_subject, test_per_subject


# get training and testing data for pose classification
def get_training_testing_data_for_pose_classification(projected, y, subjects, types, training_size):
    # init the training and testing datad
    testing_size = subjects*types - training_size

    y_train = y[:training_size]
    y_test = y[training_size:]

    training_data = projected[:training_size]
    testing_data = projected[training_size:]

    print('training_data size = ', training_size)
    print('testing_data size = ', testing_size)

    return training_data, training_size, testing_data, testing_size, y_train, y_test

# calculates the mean and covariance per class
def get_mean_and_covariance_per_class(training_data, projected, dataset):
    mu = []
    cov = []

    for i in range(0, len(training_data), 2):
        if dataset == 'face':
            mean = ((training_data[i] + training_data[i+1]) / 2).reshape(1, projected.shape[1])
            cov1 = np.matmul((training_data[i]-mean).T, training_data[i]-mean)
            cov2 = np.matmul((training_data[i+1]-mean).T, training_data[i+1]-mean)
            noise = 0.99*np.identity(cov1.shape[0])
            cov_ = (cov1 + cov2)/2 + noise
            # print(np.linalg.det(cov_))
            # break
            cov.append(cov_)
            mu.append(mean)
            if np.linalg.det(cov_) == 0:
                print('alert - zero determinant')

        else:
            for i in range(len(training_data)):
                matrix = np.array(training_data[i]['data'])
                mean = np.sum(matrix, axis=0)/matrix.shape[0]
                cov_ = (np.matmul((matrix - mean).T, (matrix-mean))) / matrix.shape[0]
                noise = 0.02*np.identity(cov_.shape[0])
                cov_ = cov_ + noise
                cov.append(cov_)
                mu.append(mean)
                if np.linalg.det(cov_) == 0 or np.linalg.det(cov_) == 0.0:
                    print('alert - zero determinant')
    return mu, cov


# Compute the mean and covariance for each training sample
def calculate_covariance_mean_knn(training_data, dataset):
    cov = []
    mu = []
    for i in range(len(training_data)):
        sample = training_data[i]
        size = sample.shape[0]
        sample = sample.reshape(1, size)
        cov_ = np.dot((sample - np.mean(sample)).T, (sample - np.mean(sample))) / size
        # add noise to make determinant non-zero
        if dataset == 'face':
            noise = 0.24*np.identity(cov_.shape[0])
        elif dataset == 'pose':
            noise = 0.03*np.identity(cov_.shape[0])
        else:
            noise = 0.01*np.identity(cov_.shape[0])
            # noise = 0.03*np.identity(cov_.shape[0])
        cov_ = cov_ + noise
        mu.append(np.mean(sample))
        cov.append(cov_)
    return cov, mu


