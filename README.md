# Different machine learning models using Breast Cancer Wisconsin (Diagnostic) Data Set from Kaggle
The goal of this repo is to test different ML models and evaluate their metrics.

## 1/Pre-processing the data

Raw data are stored in data.csv.
The Breast_cancer.py script aims to process the data, cleaning them and modifying them in a way that will help our model. The output is modified_data.csv
This dataset is a pretty good one as we have no missing values and only continuous numerical values (except for the diagnosis).


## 2/Testing a Support Vector Machine model

cancer_svm.py is a script training a SVM model. 
After training, tuning and testing it on test data, the model has converged with an F1 score of 0.97 on the CV set and a prediction accuracy of 0.96 on the test set. The best hyper-parameters for this model are C = 10000, gamma = 0.001 and a radial basis function kernel. 

After running :
    grid = GridSearchCV(estimator = my_svm, param_grid = param_grid, cv=10, verbose=2, scoring='f1_macro')
    grid.fit(X_train_scaled, y_train)
we get 
    [Parallel(n_jobs=1)]: Done 3380 out of 3380 | elapsed:   22.0s finished
    Best F1 score =  0.9734173436771627
    With best parameters :  {'C': 10000.0, 'gamma': 0.001, 'kernel': 'rbf'}
    Score on test data =  0.9649122807017544
