"""Bayes Classifer Maximum Likelihood Estimation - Subject and Poses
Returns:
    float: accuracy of the Bayes classifier
"""

import numpy as np
import matplotlib.pyplot as plt
import math
from helper_functions import *
import random

# Find the Maximum Likelihood for each subject
def BayesClassifier_Subject(testing_data, testing_size, dataset, subjects, mu, cov):

    if dataset == 'face':
        score = 0
        for i in range(testing_size):
            likelihood_list = []
            for j in range(testing_size):
                
                likelihood = (-0.5)*math.log(np.linalg.det(cov[j])) - \
                    (0.5)*np.dot( testing_data[i]-mu[j], np.dot( np.linalg.inv(cov[j]),\
                        (testing_data[i]-mu[j]).T ) )
                
                likelihood_list.append(likelihood)

            temp = np.array(likelihood_list)
            if np.argmax(temp) == i:
                score += 1
                print('Correct, score is now ', score)
            else:
                print('Incorrect Score for subject ', i)
        
        accuracy = score*100/subjects

    else:
        score = 0
        total_test_samples = len(testing_data)*testing_size
        for i in range(subjects):
            for j in range(testing_size):
                test_data = testing_data[i]['data']
                likelihood_list = []
                for k in range(subjects):
                    likelihood = (-0.5)*math.log(np.linalg.det(cov[k])) - (0.5)*np.dot( test_data[j]-mu[k], np.dot( np.linalg.inv(cov[k]), (test_data[j]-mu[k]).T ) )
                    likelihood_list.append(likelihood)
                temp = np.array(likelihood_list)
                if np.argmax(temp) == i:
                    score += 1
                    print('Correct, score is now ', score)
                else:
                    print('Incorrect Score for subject ', i)

        print('Accuracy = ', (score*100/total_test_samples))
        accuracy = score*100/total_test_samples

    return accuracy

# Bayes' classifier for pose classification
def BayesClassifier_Pose(testing_data, testing_size, y_test, cov_neutral, mu_neutral, cov_expression, mu_expression):
    score = 0

    for i in range(testing_size):
        likelihood_neutral = -0.5*math.log(np.linalg.det(cov_neutral)) - \
            0.5*np.dot((testing_data[i] - mu_neutral), np.dot(np.linalg.inv(cov_neutral), \
                (testing_data[i] - mu_neutral).T))
    
        likelihood_expression = -0.5*math.log(np.linalg.det(cov_expression)) \
            - 0.5*np.dot((testing_data[i] - mu_expression), np.dot(np.linalg.inv(cov_expression), \
                (testing_data[i] - mu_expression).T))
        
        if likelihood_neutral > likelihood_expression:
            predicted_class = 1
        else:
            predicted_class = -1
        if predicted_class == y_test[i]:
            score += 1
            # print('Correct classification, score = ', score)
        # else:
            # print('Incorrect classification for test data ', i)

    accuracy = score*100/testing_size
    return accuracy