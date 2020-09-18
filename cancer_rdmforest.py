import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV, learning_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, plot_confusion_matrix

df = pd.read_csv("modified_data.csv")

#Creating Inputs and Results
X = df.drop(["diagnosis"], axis = 1)
y = df["diagnosis"]

#Splitting between train and test, using shuffle and seeding for reproductibility
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 5)

#Features scaling using a MinMax scaling, first on X_train then on X_test
my_scaler = MinMaxScaler()
X_train_scaled = my_scaler.fit_transform(X_train)
X_test_scaled = my_scaler.fit_transform(X_test)

#Creating our RandomForest classifier
my_forest = RandomForestClassifier()

#Creating our GridSearch object that will test our estimator with a set of different hyper-parameters
#As the RandoForest has a lot of hyperparameters to optimize, we will use a RandomGridSearch that will not try all the combination

# #Nbr of trees in the forest
# n_estimators = [int(x) for x in np.linspace(200, 2000, 10)]
# #Nbr of features to consider at every split
# max_features = ["auto", "sqrt"]
# #Maximum level number of levels in the tree
# max_depth = [int(x) for x in np.linspace(10, 110, 11)]
# max_depth.append(None)
# #Minimum number of samples required to split a node
# min_samples_split = [2, 5, 10]
# #Minimum number of samples requiered at each leaf node
# min_samples_leaf = [1, 2, 4]
# #Method of selecting samples for training each tree
# bootstrap = [True, False]

# #Creating the random grid
# param_random_grid = {"n_estimators" : n_estimators,
#                 "max_features" : max_features,
#                 "max_depth" : max_depth,
#                 "min_samples_split" : min_samples_split,
#                 "min_samples_leaf" : min_samples_leaf,
#                 "bootstrap" : bootstrap}

# grid = RandomizedSearchCV(estimator=my_forest, param_distributions=param_random_grid, cv=10, verbose=2, n_iter=100, scoring="f1_macro")

# #Fitting our GridSearch
# grid.fit(X_train_scaled, y_train)

# #Print ou best score for the best parameters
# print("Best F1 score = ", grid.best_score_)
# print("With best parameters : ", grid.best_params_)

# #Saving our best estimator (RandomForest) as our model and applying it to test data to compute the final score
# model = grid.best_estimator_
# predictions = model.predict(X_test_scaled)
# print("Score on test data = ", model.score(X_test_scaled, y_test))


#Creating our GridSearch object that will test our estimator with a set of different hyper-parameters
param_grid = {"n_estimators" : [150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250],
                "max_features" : ["auto", "sqrt"],
                "max_depth" : [40, 45, 50, 55, 60],
                "min_samples_split" : [2, 3],
                "min_samples_leaf" : [1, 2],
                "bootstrap" : [True, False]}
grid = GridSearchCV(estimator = my_forest, param_grid = param_grid, cv=10, verbose=2, scoring='f1_macro', n_jobs=-2)

#Fitting our GridSearch
grid.fit(X_train_scaled, y_train)

#Print ou best score for the best parameters
print("Best F1 score = ", grid.best_score_)
print("With best parameters : ", grid.best_params_)

#Saving our best estimator (RandomForest) as our model and applying it to test data to compute the final score
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