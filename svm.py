from sklearn.model_selection import GridSearchCV
from sklearn import svm
from pathlib import Path
from utilities import plot_grid_search, evaluation_metrics
import json
import time

def train_default(X_train, X_test, y_train, y_test, folder_name, kernel = None):
    """
    Trains a SVM using the default parameters and the given kernel.
    :param X_train: The training features.
    :param X_test: The testing features.
    :param y_train: The training classes.
    :param y_test: The testing classes.
    :param folder_name: The name of the folder.
    :param kernel: (Optional) The kernel to be used. In case
    none is provided, 'rbf' will be used.
    """
    Path(f"figures/{folder_name}").mkdir(parents = True, exist_ok = True)
    
    model = default_svm(X_train, y_train, kernel = kernel)
    
    # Predicting the test results.
    y_pred = model.predict(X_test)
    
    # Printing the metrics.
    confusion_matrix = evaluation_metrics(y_test, y_pred, folder_name)
    
    return confusion_matrix

def train_grid_search(X_train, X_test, y_train, y_test, grid_parameters, \
                      folder_name):
    """
    Finds the optimal parameters by performing grid search
    using the given parameters.
    :param X_train: The training features.
    :param X_test: The testing features.
    :param y_train: The training classes.
    :param y_test: The testing classes.
    :param grid_parameters: The grid search parameters.
    :param folder_name: The name of the folder.
    """
    Path(f"figures/{folder_name}").mkdir(parents = True, exist_ok = True)
    
    start = time.time()

    grid = svm_grid_search(X_train, y_train, grid_parameters)
    best_params = grid.best_params_
    
    # Predicting the test results.
    y_pred = grid.predict(X_test)
    
    # Printing the metrics.
    confusion_matrix = evaluation_metrics(y_test, y_pred, folder_name)
    
    if "rbf" in grid_parameters["kernel"]:
        plot_grid_search(grid.cv_results_, "gamma", "C", folder_name)
    elif "poly" in grid_parameters["kernel"]:
        plot_grid_search(grid.cv_results_, "degree", "C", folder_name)
        
    end = time.time()
    
    results = {
        "best_params": best_params,
        "total_time": end - start
    }
        
    # Store best params in json file
    with open(f"figures/{folder_name}/grid.json", "w") as grid:
        json.dump(results, grid)
    
    return confusion_matrix

def svm_grid_search(X_train, y_train, grid_parameters):
    """
    Performs grid search on a Support Vector Machine
    :param X_train: The training features.
    :param y_train: The training classes.
    :grid_parameters: The grid search parameters.
    :returns: The grid object.
    """
    grid = GridSearchCV(svm.SVC(), grid_parameters, refit = True, \
                        verbose = 5, n_jobs = -1, scoring = "precision_macro",\
                        return_train_score = True)
    
    
    grid.fit(X_train, y_train)
    
    print(grid.best_params_)
    
    return grid

def default_svm(X_train, y_train, kernel = None):
    """
    Creates a SVM model with the default parameters.
    :param X_train: The training features.
    :param y_train: The training classes.
    :param kernel: (Optional) The kernel to be used. In case
    none is provided, 'rbf' will be used.
    """
    model = None
    if kernel is not None:
        model = svm.SVC(kernel = kernel)
    else:
        model = svm.SVC()
    
    model.fit(X_train, y_train)
    
    return model
    