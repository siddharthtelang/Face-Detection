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

def svm_classification(training_data, testing_data, y_train, y_test, kernel_type = 'rbf', reduction='pca'):

    param_lookup = {'rbf':6, 'poly':1, 'linear':1}
    # params = [1,2,3,4,5,6,7,8,9,10]
    params = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,2,3,4,5]
    scores = []
    param = param_lookup.get(kernel_type, 6)

    for C in params:
        training_size = training_data.shape[0]
        testing_size = testing_data.shape[0]
        
        threshold = 1e-5
        kernel = get_kernel(training_data, kernel_type=kernel_type, param=param)

        sv = solve_dual_optimization(kernel, y_train, training_size, threshold=threshold, C=C)

        # Find the intercept
        idx = np.argsort(sv)
        idx = idx[::-1][0]
        k1 = evaluate_kernel(training_data, training_data[idx], kernel_type=kernel_type, param=param)
        sum = 0.0
        for m in range(training_size):
            sum += k1[m]*sv[m]*y_train[m]
        theta0 = 1/y_train[idx] - sum

        # testing
        score = 0
        for i in range(len(testing_data)):

            k2 = evaluate_kernel(training_data, testing_data[i], kernel_type=kernel_type, param=param)
            test = 0.0

            for m in range(training_size):
                test += k2[m]*y_train[m]*sv[m]

            test += theta0

            if (test*y_test[i] >= 0):
                score += 1
                # print('Correct')
            # else:
                # print('Incorrect')

        print(score)
        accuracy = (score/testing_size)*100
        scores.append(accuracy)
        print('Accuracy with ', kernel_type, ' kernel with param ', str(param), ' vs Soft margin', str(C), ' is = ', str(accuracy))

    label = kernel_type + ' parameter = ' + str(param) +  ' and  Soft margin Slack parameter'
    plt.title('SVM Kernel Classification')
    plt.xlabel(label)
    plt.ylabel('Accuracy')
    plt.plot(np.array(params), np.array(scores))
    plt.show()
    

def classify_pose_SVM(dataset_file, subjects, types, kernel_type='rbf',\
                      params=6, usePCA=True, useMDA=False, training_size=300):

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

    svm_classification(training_data, testing_data, y_train, y_test,\
                       kernel_type = kernel_type, reduction=reduction)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', '--set',required=False, default='Data/data.mat', type=str)
    parser.add_argument('-subjects', '--sub',required=False, default='200', type=int)
    parser.add_argument('-types', '--type',required=False, default='2', type=int)
    parser.add_argument('-trainingSize', '--trsize',required=False, default='300', type=int)
    parser.add_argument('-kernel', '--ker',required=False, default='rbf', type=str)
    parser.add_argument('-param', '--gamma',required=False, default='6', type=int)
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
    classify_pose_SVM(dataset_file, subjects, types, kernel_type,\
                      params, usePCA, useMDA, training_size)