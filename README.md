# Different machine learning models using Breast Cancer Wisconsin (Diagnostic) Data Set from Kaggle
The goal of this repo is to test different ML models and evaluate their metrics.

## 1/Pre-processing the data

The Breast_cancer.py script aims to process the data, cleaning them and modifying them in a way that will help our model.
This dataset is a pretty good one as we have no missing values and only continuous numerical values (except for the diagnosis).


## 2/Testing a Support Vector Machine model

cancer_svm.py is a script training a SVM model. 
After training, tuning and testing it on test data, the model has converged with an F1 score of 0.97 on the CV set and a prediction accuracy of 0.96 on the test set. The best hyper-parameters for this model are C = 10000, gamma = 0.001 and a radial basis function kernel. 
