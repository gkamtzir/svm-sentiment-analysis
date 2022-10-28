import pandas as pd
from utilities import preprocess_data
from svm import train_grid_search, train_default
from knn import train_grid_search as knn_train_grid_search
from ncc import train_grid_search as ncc_train_grid_search

# Importing dataset.
data = pd.read_csv("tweet_emotions.csv")

X_train, X_test, y_train, y_test = preprocess_data(data, 10, True)

# Default SVM (RBF).
train_default(X_train, X_test, y_train, y_test, "rbf_default")

# GridSearch on Linear kernel.
grid_parameters = {"C": [0.001, 0.01, 0.1, 1, 10, 100],
    "kernel": ["linear"]}
train_grid_search(X_train, X_test, y_train, y_test, grid_parameters, "linear")

# GridSearch on RBF kernel.
grid_parameters = {"C": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
    "gamma": [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000],
    "kernel": ["rbf"]}
train_grid_search(X_train, X_test, y_train, y_test, grid_parameters, "rbf")

# GridSearch on RBF kernel.
grid_parameters = {"C": [1180, 1182, 1184, 1186, 1186],
    "gamma": [0.075, 0.076, 0.077, 0.078, 0.079, 0.08, 0.081, 0.082, 0.083, 0.084, 0.085],
    "kernel": ["rbf"]}
train_grid_search(X_train, X_test, y_train, y_test, grid_parameters, "rbf_target")

# GridSearch on Poly kernel.
grid_parameters = {"C": [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000],
    "degree": [2, 3, 4],
    "kernel": ["poly"]}
train_grid_search(X_train, X_test, y_train, y_test, grid_parameters, "poly")
