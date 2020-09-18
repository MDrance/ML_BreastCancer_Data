# Different machine learning models on Breast Cancer Wisconsin (Diagnostic) Data Set from Kaggle
The goal of this repository is to test different ML models and evaluate their metrics.

## 1.Pre-processing the data

Raw data are stored in **data.csv**.
The **Breast_cancer.py** script aims to process the data, cleaning them and modifying them in a way that will help our model. The output is **modified_data.csv**
This dataset is a pretty good one as we have no missing values and only continuous numerical values (except for the diagnosis).


## 2.Testing a Support Vector Machine model

**cancer_svm.py** is a script training and optimizing our SVM model. 
For the fine tuning part, we use cross-validation and grid-search methods to find the best value for C (regularization parameter), the best kernel bewteen polynomial and Radial Basis Function and the best value for gamma(kernel coeficient, curvature of decision boundary).

After running :
```
grid = GridSearchCV(estimator = my_svm, param_grid = param_grid, cv=10, verbose=2, scoring='f1_macro')
grid.fit(X_train_scaled, y_train)
```
we get :
```
[Parallel(n_jobs=1)]: Done 3380 out of 3380 | elapsed:   22.0s finished
Best F1 score =  0.9734173436771627
With best parameters :  {'C': 10000.0, 'gamma': 0.001, 'kernel': 'rbf'}
Score on test data =  0.9649122807017544
```
Our learning curves and confusion matrix look like this :
![Learning curves](https://github.com/MDrance/ML_BreastCancer_Data/blob/master/confmatlearcur.png)

* The confusion matrix shows no false positives and 4 false negatives.
* We observe from our learning curves that the Train and Validation Curves converge at a score of 0.98 when the train size reaches 325 samples. It is not clear if adding more data could help them converging more (getting a lower irreductible error). 
* As our training curves converge and converge to a high score value, our model is not suffering high bias (underfitting) or high variance (overfitting).

## 3.Testing a Random Forest Classifier

**cancer_rdmforest.py** is a script training and optimizing a Random Forest classifier.
For the fine tuning part, we use crosse-validation, random grid-search and grid-search methods. Random Forest has a lot more hyper-parameters to optimize :
* Number of trees in the forest
* Number of features to consider at every split
* Maximum number of levels in tree
* Maximum number of samples required to split a node
* Minimum number of samples required at each leaf node
* Method of selecting samples for training each tree
As it would make a huge number of possible combinations, we first use a RandomizedSearchCV method to evaluate ranges for our hyper-parameters, then applying a GridSearchCV with fewer possibilities to evaluate our model and plot our metrics.
After running :
```
grid = RandomizedSearchCV(estimator=my_forest, param_distributions=param_random_grid, cv=10, verbose=2, n_iter=100, scoring="f1_macro", n_jobs = -2)
grid.fit(X_train_scaled, y_train)
```
we get :
```
[Parallel(n_jobs=1)]: Done 1000 out of 1000 | elapsed: 26.7min finished
Best F1 score =  0.9588129275206505
With best parameters :  {'n_estimators': 200, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'max_depth': 50, 'bootstrap': False}
Score on test data =  0.9736842105263158
```
We then run our GridSearchCV to obtain our metrics. 
After running :
```
grid = GridSearchCV(estimator = my_forest, param_grid = param_grid, cv=10, verbose=2, scoring='f1_macro', n_jobs=-2)
grid.fit(X_train_scaled, y_train)
```
we get :
```
[Parallel(n_jobs=-2)]: Done 8800 out of 8800 | elapsed: 13.0min finished
Best F1 score =  0.9614443577071965
With best parameters :  {'bootstrap': False, 'max_depth': 40, 'max_features': 'sqrt', 'min_samples_leaf': 2, 'min_samples_split': 3, 'n_estimators': 200}
Score on test data =  0.9736842105263158
```
The score on test data remains the same but we got a slightly better F1-score.
Our learning curves and confusion matrix look like this :
![Learning curves](https://github.com/MDrance/ML_BreastCancer_Data/blob/master/forestmetrics.png)

* The confusion matrix shows no false positives and 3 false negatives, showing better results than the SVM model.
* We observe from our learning curves that the Validation score increases and it is not clear if it could increase more (no real plateau). Adding more data
could help observe if the train phase reaches its limit. Train curve is not important as in Random Forest, samples take the same path through the trees when
training and predicting.
* Globally, the metrics are a little bit better than the ones using an SVM model and our model is not suffering high bias (underfitting) or high variance (overfitting).
