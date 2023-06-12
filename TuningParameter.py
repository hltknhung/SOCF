import pandas as pd
import numpy as np
import os
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from pyGRNN import GRNN
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import make_scorer
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
import time

p = os.path.abspath('..')
file_path_train = os.path.join(p + '/validation/DS1/' + 'training.csv')
df_train = pd.read_csv(file_path_train)

def tuningSVR(X, y):
    t0 = time.time()
    X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.3)
    X_validation = preprocessing.minmax_scale(X_validation)
    parameters = [
        {'kernel': ['rbf', 'poly', 'linear'],
         'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
         'C': [5, 10, 50, 100, 150],
         'epsilon': [1, 0.1, 0.01, 0.001, 0.0001],
         'max_iter': [100000]
         }]
    print("Tuning hyper-parameters")
    scorer = make_scorer(mean_squared_error, greater_is_better=True)
    Reg = GridSearchCV(SVR(), parameters, cv=5, scoring=scorer)
    Reg.fit(X_validation, y_validation)
    print("Training time:", round((time.time() - t0) * 1000, 3))
    return Reg.best_params_


def tuningMLP(X, y):
    t0 = time.time()
    X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.3)
    X_validation = preprocessing.minmax_scale(X_validation)
    parameters = [
        {'hidden_layer_sizes': [(5, 1), (6, 1), (7, 1), (8, 1)],
         'max_iter': [10000],
         'learning_rate_init': [0.01, 0.02, 0.03, 0.04, 0.05],
         # 'learning_rate': ['constant', 'invscaling', 'adaptive'],
         # 'activation': ['identity', 'logistic', 'tanh', 'relu'],
         # 'solver': ['lbfgs', 'sgd', 'adam'],
         'alpha': [0.1, 0.01, 0.001, 0.0001, 0.00001],
         'momentum': [0.1, 0.2, 0.3, 0.4, 0.5]
         }]
    print("Tuning hyper-parameters")
    scorer = make_scorer(mean_squared_error, greater_is_better=True)
    Reg = GridSearchCV(MLPRegressor(), parameters, scoring=scorer)
    Reg.fit(X_validation, y_validation)
    print("Training time:", round((time.time() - t0) * 1000, 3))
    Reg.best_params_


def tuningRF(X, y):
    t0 = time.time()
    X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.3)
    X_validation = preprocessing.minmax_scale(X_validation)
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
    # Number of features to consider at every split
    # max_features = ['sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
    # max_depth.append(None)
    # Minimum number of samples required to split a node
    # min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    # bootstrap = [True, False]
    parameters = [
        {'n_estimators': n_estimators,
         # 'max_features': max_features,
         'max_depth': max_depth,
         # 'min_samples_split': min_samples_split,
         'min_samples_leaf': min_samples_leaf,
         # 'bootstrap': bootstrap
         }
    ]
    print("Tuning hyper-parameters")
    scorer = make_scorer(mean_squared_error, greater_is_better=True)
    Reg = GridSearchCV(RandomForestRegressor(), parameters, cv=5, scoring=scorer)
    Reg.fit(X_validation, y_validation)
    print("Training time:", round((time.time() - t0) * 1000, 3))
    Reg.best_params


def tuningDT(X, y):
    t0 = time.time()
    X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.3)
    X_validation = preprocessing.minmax_scale(X_validation)
    parameters = [
        {"splitter": ["best", "random"],
         "max_depth": [1, 3, 5, 7, 9, 11, 12],
         "min_samples_leaf": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
         "min_weight_fraction_leaf": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
         "max_features": ["auto", "log2", "sqrt", None],
         "max_leaf_nodes": [None, 10, 20, 30, 40, 50, 60, 70, 80, 90]}]

    print("Tuning hyper-parameters")
    scorer = make_scorer(mean_squared_error, greater_is_better=True)
    Reg = GridSearchCV(DecisionTreeRegressor(), parameters, cv=5, scoring=scorer)
    Reg.fit(X_validation, y_validation)
    print("Training time:", round((time.time() - t0) * 1000, 3))
    Reg.best_params_


def tuningGB(X, y):
    t0 = time.time()
    X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.3)
    X_validation = preprocessing.minmax_scale(X_validation)
    parameters = [
        {"n_estimators": range(20, 81, 10),
         "max_depth": range(5, 16, 2),
         # "min_samples_split": range(200,1001,200),
         "min_samples_leaf": range(30, 71, 10),
         # "max_features": range(7,20,2),
         # "subsample": [0.6,0.7,0.75,0.8,0.85,0.9]
         }]
    print("Tuning hyper-parameters")
    scorer = make_scorer(mean_squared_error, greater_is_better=True)
    Reg = GridSearchCV(GradientBoostingRegressor(), parameters, cv=5, scoring=scorer)
    Reg.fit(X_validation, y_validation)
    print("Training time:", round((time.time() - t0) * 1000, 3))
    return Reg.best_params_


def tuningKNN(X, y):
    t0 = time.time()
    X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.3)
    X_validation = preprocessing.minmax_scale(X_validation)
    parameters = [{"n_neighbors": list(range(2, 15))}]
    print("Tuning hyper-parameters")
    scorer = make_scorer(mean_squared_error, greater_is_better=True)
    Reg = GridSearchCV(KNeighborsRegressor(), parameters, cv=5, scoring=scorer)
    Reg.fit(X_validation, y_validation)
    print("Training time:", round((time.time() - t0) * 1000, 3))
    return Reg.best_params_
