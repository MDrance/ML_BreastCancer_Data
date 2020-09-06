# Different machine learning models on Breast Cancer Wisconsin (Diagnostic) Data Set from Kaggle
The goal of this repository is to test different ML models and evaluate their metrics.

## 1.Pre-processing the data

Raw data are stored in **data.csv**.
The **Breast_cancer.py** script aims to process the data, cleaning them and modifying them in a way that will help our model. The output is **modified_data.csv**
This dataset is a pretty good one as we have no missing values and only continuous numerical values (except for the diagnosis).


## 2.Testing a Support Vector Machine model

**cancer_svm.py** is a script training and optimizing our SVM model. 
For the fine tuning part, we use cross-validation and grid-search methods to find the best value for C (regularization), the best kernel bewteen polynomial and Radial Basis Function and if using RBF the best value for gamma(curvature of decision boundary).

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
