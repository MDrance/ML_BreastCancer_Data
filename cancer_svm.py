import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, plot_confusion_matrix

df = pd.read_csv("modified_data.csv")

#Creating Inputs and Results
X = df.drop(["diagnosis"], axis=1)
y = df["diagnosis"]


#Splitting between train and test, using shuffle and seeding for reproductibility
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 5)

#Features scaling using a MinMax scaling, first on X_train then on X_test
my_scaler = MinMaxScaler()
X_train_scaled = my_scaler.fit_transform(X_train)
X_test_scaled = my_scaler.fit_transform(X_test)

#Creating our SVM model
my_svm = SVC()

#Creating our GridSearch object that will test our estimator with a set of different hyper-parameters
test_values = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1, 10, 1e2, 1e3, 1e4, 1e5]
param_grid = {"kernel" : ["poly", "rbf"], "C" : test_values, "gamma" : test_values}
grid = GridSearchCV(estimator = my_svm, param_grid = param_grid, cv=10, verbose=2, scoring='f1_macro')

#Fitting our GridSearch
grid.fit(X_train_scaled, y_train)

#Print ou best score for the best parameters
print("Best F1 score = ", grid.best_score_)
print("With best parameters : ", grid.best_params_)

#Saving our best estimator (SVM) as our model and applying it to test data to compute the final score
model = grid.best_estimator_
predictions = model.predict(X_test_scaled)
print("Score on test data = ", model.score(X_test_scaled, y_test))

#Creating the confusion matrix to have a look at the results
fig, axs = plt.subplots(ncols=2, figsize = (10,4))
plot_confusion_matrix(model, X_test_scaled, y_test, ax=axs[0])

#Creating the learning curves to see if our model could use more data
N, train_score, val_score = learning_curve(model, X_train_scaled, y_train, train_sizes = np.linspace(0.1, 1.0, 10), cv=5)
sns.lineplot(N, train_score.mean(axis=1), label = "Train", ax=axs[1])
sns.lineplot(N, val_score.mean(axis=1), label = "Validation", ax=axs[1])
plt.show()

"""
We observe from our learning curves that the Train and Validation Curves converge at a score of 0.98 when the train size reach 325 samples
It is not clear if adding more data could help them ending closer
As our training curves converge and converge to a high score value, our model is not suffering high bias (underfitting) or high variance (overfitting)
"""