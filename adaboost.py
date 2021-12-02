from svm_helper import *
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.decomposition import PCA
from helper_functions import *
from my_pca import *
from my_lda import *
import argparse
    
def boosted_svm_classification(training_data, testing_data, y, projected,\
                       kernel_type = 'linear', reduction='pca', iterations=10):


    training_size = training_data.shape[0]
    testing_size = testing_data.shape[0]
    validation_data = projected[:training_size]
    y_validation = y[:training_size]
    y_test = y[training_size:]
    # weight vector
    w = np.zeros(shape=(iterations, training_size))
    P = np.zeros(shape=(iterations, training_size))
    # coefficient multipliers
    a = np.zeros(shape=(iterations))
    test = np.zeros(shape=training_size)
    # init the initial weight with equal probability
    w[0] = 1/training_size*np.ones(training_size)
    theta0 = np.zeros(shape=(iterations))
    theta = np.zeros(shape=(iterations, training_data.shape[1])) 

    for k in range(iterations-1):

        threshold = 1e-5

        # calculate P for each sample
        for i in range(training_size-1):
            P[k][i] = w[k][i] / (np.sum(w[k]))

        # training size for weak classifier
        training_size_weak = 0
        error = 0.0

        while(True):

            ei = np.zeros(shape=(training_size))

            # update the training on weak classifer till the error falls below 50%
            training_size_weak += 20
            # get the test, train, validation data
            training_data, validation_data, testing_data, y_train,\
                y_validation, y_test = get_test_train_validation_adaboost\
                    (subjects, types, projected, y, training_size_weak, testing_size)

            kernel = get_kernel(training_data, kernel_type=kernel_type)

            # find the multiplers
            sv = solve_dual_optimization(kernel, y_train, training_size_weak)

            # hyperplane
            theta[k] = np.dot((sv*y_train).T, training_data)

            # Find the intercept
            idx = np.argsort(sv)
            idx = idx[::-1][0]

            theta0[k] = 1/y_train[idx] - np.dot(theta[k].T, training_data[idx])

            # cross-validation
            error = 0.0
            for i in range(len(validation_data)):
                test[i] = np.dot(theta[k].T, validation_data[i]) + theta0[k]

                if (test[i]*y_validation[i] < 0):
                    error += P[k][i]
            
            print('Error in iteration ', str(k+1), ' is = ', str(error))

            # weak classifier
            if error < 0.5:
                break
        
        # update the coefficient
        if error != 0:
            a[k] = 0.5*math.log((1-error)/error)

        # update next weights
        for i in range(training_size):
            w[k+1][i] = w[k][i] * math.exp( (-1*y_validation[i]*test[i]) )
        
    accuracy_training = get_accuracy(validation_data, iterations,\
                            theta, theta0, y_validation, a)

    accuracy_testing = get_accuracy(testing_data, iterations,\
                            theta, theta0, y_test, a)
    
    print('Total Iterations =  ', iterations)
    print('Training Accuracy =', accuracy_training)
    print('Testing Accuracy =', accuracy_testing)

    return accuracy_training, accuracy_testing
    
    


def classify_boosted_SVM(dataset_file, subjects, types, kernel_type='linear',\
                         params=6, usePCA=True, useMDA=False, training_size=300,\
                         iterations=5):

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

    if usePCA:
        reduction='pca'
    else:
        reduction='lda'

    accuracy_training, accuracy_testing =\
    boosted_svm_classification(training_data, testing_data, y, projected,\
                       kernel_type = kernel_type, reduction=reduction,\
                       iterations=iterations)
    return accuracy_training, accuracy_testing



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', '--set',required=False, default='Data/data.mat', type=str)
    parser.add_argument('-subjects', '--sub',required=False, default='200', type=int)
    parser.add_argument('-types', '--type',required=False, default='2', type=int)
    parser.add_argument('-trainingSize', '--trsize',required=False, default='300', type=int)
    parser.add_argument('-kernel', '--ker',required=False, default='linear', type=str)
    parser.add_argument('-param', '--gamma',required=False, default='6', type=int)
    parser.add_argument('-iterations', '--iter',required=False, default='20', type=int)
    parser.add_argument('-pca', '--pca',required=False, default=False, type=bool)
    parser.add_argument('-mda', '--mda',required=False, default=False, type=bool)
    args = vars(parser.parse_args())
    dataset_file = args['set']
    subjects = args['sub']
    types = args['type']
    usePCA = args['pca']
    useMDA = args['mda']
    training_size = args['trsize']
    kernel_type = args['ker']
    params = args['gamma']
    iter = args['iter']
    atr = []
    ate = []
    iterations = [i for i in range(1,iter)]

    for iter in iterations:
        x,y = classify_boosted_SVM(dataset_file, subjects, types, kernel_type=kernel_type,\
                        params=params, usePCA=True, useMDA=False, training_size=300,\
                        iterations=iter)
        atr.append(x)
        ate.append(y)

    # plot the results

    plt.figure()
    plt.title('Boosted Linear SVM - Training')
    plt.xlabel('Iterations')
    plt.ylabel('Training Accuracy')
    plt.plot(iterations, atr)

    plt.figure()
    plt.title('Boosted Linear SVM - Testing')
    plt.xlabel('Iterations')
    plt.ylabel('Testing Accuracy')
    plt.plot(iterations, ate)

    plt.show()

