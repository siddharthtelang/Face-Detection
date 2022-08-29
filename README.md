# Face-Detection
Face detection (class and psoe) using various Classifiers.
Dimentionality Reduction using PCA and MDA

## Classifiers implemented from scratch
- Bayes
- kNN
- SVM (Linear, RBF, Polynomial kernels)
- AdaBoost with SVM

## Author

### Siddharth Telang (stelang@umd.edu)

## Subject Code
### CMSC828C/ENEE633 Project 1

## Programming language used: Python3+
### Dependencies (to be installed through pip):
```
1) sklearn(used only for PCA): pip install sklearn
2) matplotlib: pip install matplotlib
3) numpy: pip install numpy
4) scipy (.mat to python): pip install scipy
5) cvxopt (quadratic solver): pip install cvxopt
```

## Contents:
```
1) Code files:
- Helper functions (my_pca, my_mda, my_lda, helper_functions, svm_helper) used by main files
- bayes_classifier_subject, knn_subject: Subject label classification
- bayes_classifier: bayes' classifier implementation
- classify_pose_Bayes, classify_pose_KNN: Pose identification for Data set 1
- svm_classifier, adaboost: Pose identification for Data set 1
2) Report
3) Figures - all plots
4) Data folder containing the dataset
```

## Steps to run the code:

- Please ensure this to be the current working directory.
- Various commands with different permutations are mentioned below.
- You may use this on the command prompt and terminal.
- A choice of choosing among pca or mda is provided. Feel free to update if required, only one of them can be set to True at a time.
- trainingSize parameter can be altered to test on various training and testing size
- kernel parameter is provided to select between 'rbf', 'poly', and 'linear' kerel svm
- iterations parameter can be updated for more number of iterations in boosted svm

### 1) Subject Label classification
- Bayes Classifier
```
python bayes_classifier_subject.py -dataset Data/data.mat -subjects 200 -types 3 -pca True
python bayes_classifier_subject.py -dataset Data/pose.mat -subjects 68 -types 13 -mda True
python bayes_classifier_subject.py -dataset Data/illumination.mat -subjects 68 -types 21 -mda True
```
- k-NN
```
python knn_subject.py -dataset Data/data.mat -subjects 200 -types 3 -pca True
python knn_subject.py -dataset Data/pose.mat -subjects 68 -types 13 -mda True
python knn_subject.py -dataset Data/illumination.mat -subjects 68 -types 21 -mda True
```

### 2) Neutral vs Expression identification

- Bayes Classifier
```
python classify_pose_Bayes.py -dataset Data/data.mat -subjects 200 -types 2 -trainingSize 200 -pca True
python classify_pose_Bayes.py -dataset Data/data.mat -subjects 200 -types 2 -trainingSize 200 -mda True
python classify_pose_Bayes.py -dataset Data/data.mat -subjects 200 -types 2 -trainingSize 300 -mda True
```

- k-NN
```
python classify_pose_KNN.py -dataset Data/data.mat -subjects 200 -types 2 -trainingSize 200 -pca True
python classify_pose_KNN.py -dataset Data/data.mat -subjects 200 -types 2 -trainingSize 200 -mda True
python classify_pose_KNN.py -dataset Data/data.mat -subjects 200 -types 2 -trainingSize 100 -mda True
```
- Kernel SVM
```
python svm_classifier.py -dataset Data/data.mat -subjects 200 -types 2 -trainingSize 300 -pca True -kernel rbf 
python svm_classifier.py -dataset Data/data.mat -subjects 200 -types 2 -trainingSize 300 -mda True -kernel rbf 
python svm_classifier.py -dataset Data/data.mat -subjects 200 -types 2 -trainingSize 300 -pca True -kernel poly
python svm_classifier.py -dataset Data/data.mat -subjects 200 -types 2 -trainingSize 300 -pca True -kernel linear
```
- Ada-Boost (Linear SVM)
```
python adaboost.py -dataset Data/data.mat -subjects 200 -types 2 -trainingSize 300 -pca True -kernel linear - iterations 10
```
