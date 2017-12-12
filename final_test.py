#!/usr/bin/env python
"""For testing Wine dataset"""

# import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.svm import SVR
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import metrics
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
import matplotlib.pyplot as plt
import itertools
import pandas as pd
import numpy as np
import time

import warnings
warnings.filterwarnings("ignore")

def main():
    """Processes the data and calls test functions"""
    data = dict()
    test = dict()
    scaler = MaxAbsScaler()

    test = {'training' : None, 'testing' : None, 'param_grid' : None, 'scorer' : None}

    param_grid = {'C': 0, # Later this gets set based on data.
                  'epsilon': np.linspace(0, 5, 32),
                  'gamma': np.logspace(-10, 9, 32, base=2)}

    # Process red wine data
    red_data = pd.read_csv("winequality-red.csv")
    samples, target = process_wine_data(red_data, scaler=scaler)
    param_grid['C'] = [best_C_parameter(target)]
    test = {**test, **samples}
    test['param_grid'] = param_grid
    test['scorer'] = metrics.make_scorer(accuracy_score, tolerance=1.5)
    data["red-max-abs-scaled"] = test
    # training, testing, _ = process_wine_data(red_data)
    # test['training'] = training
    # test['testing'] = testing
    # data["red-unscaled"] = test

    # Process white wine data
    white_data = pd.read_csv("winequality-white.csv", sep=';')
    samples, target = process_wine_data(red_data, scaler=scaler)
    param_grid['C'] = [best_C_parameter(target)]
    test = {**test, **samples}
    test['param_grid'] = param_grid
    data["white-max-abs-scaled"] = test
    # training, testing, _ = process_wine_data(red_data)
    # test['training'] = training
    # test['testing'] = testing
    # data["white-unscaled"] = test

    # # Combine and process both wine data
    # combined_data = pd.concat([red_data, white_data])
    # combined_data = shuffle(combined_data)
    # features, target = process_wine_data(combined_data)
    # data["combined-unscaled"] = (features, target)
    # scaler.fit(features)
    # features = scaler.transform(features) # Scale features to be similar
    # data["combined-max-abs-scaled"] = (features, target)
    # showData(data)
    data = parameter_test(data)
    quality_prediction(data)
    # showData(data)

def parameter_test(data):
    """Calls best_SVR_parameters() on each data."""
    for key in data:
        data[key]['bestparams'] = best_SVR_parameters(data[key])

    return data

def best_SVR_parameters(data):
    """Returns best parameters for SVR."""

    estimator = SVR(max_iter=50000, kernel='rbf')
    grid = GridSearchCV(estimator, param_grid=data['param_grid'], scoring=data['scorer'], n_jobs=4)
    start = time.time()
    grid.fit(data['training']['x'], data['training']['y'])
    print("Time: " + str(time.time() - start))
    print("Best Parameters: " + str(grid.best_params_))
    print("Score: " + str(grid.best_score_))
    return grid.best_params_

def quality_prediction(data):
    """Outputs the mean absolute error and the confusion matrix"""    
    for key in data:
        train = data[key]['training']
        best_params = data[key]['bestparams']
        estimator = SVR(**best_params, kernel='rbf')
        estimator.fit(train['x'], train['y'])

        test = data[key]['testing']
        prediction = estimator.predict(test['x'])

        mae = metrics.mean_absolute_error(test['y'], prediction)
        tolerance = 0.5
        t_pred = tolerance_prediction(test['y'], prediction, tolerance)
        conf = metrics.confusion_matrix(test['y'], t_pred, labels=range(3,10))

        mae_str = str(mae)
        print(key)
        print('mean absolute error = ' + mae_str)
        plt.clf()
        plot_confusion_matrix(conf, range(3,10), normalize=False,
                              title=key + '\nmean absolute error = ' + mae_str)
        plt.savefig(key + "-cm", dpi=150)
        plt.clf()
        plot_confusion_matrix(conf, range(3,10), normalize=True,
                              title=key + '\nmean absolute error = ' + mae_str)
        plt.savefig(key + "-cmnorm", dpi=150)

def process_wine_data(data, testsize=0.30, scaler=None):
    """Returns features and target of wine data."""
    features = data.loc[:, "fixed acidity":"alcohol"].as_matrix()
    target = data.loc[:, "quality"].as_matrix()
    if scaler:
        scaler.fit(features)
        features = scaler.transform(features) # Scale features to be similar
    x_train, x_test, y_train, y_test = train_test_split(features, target,
                                                        test_size=testsize)
    samples = {'training' : {'x' : x_train, 'y' : y_train},
               'testing' : {'x' : x_test, 'y' : y_test} }
    return samples, target

def best_C_parameter(target):
    mean = np.mean(target)
    std = np.std(target)

    value1 = np.abs(mean + (3 * std))
    value2 = np.abs(mean - (3 * std))

    return max(value1, value2)

def tolerance_prediction(y_values, predictions, tolerance = 0.5):
    result = list()
    for val, pred in zip(y_values, predictions):
        if np.abs(val - pred) < tolerance:
            result.append(val)
        else:
            result.append(np.around(pred))

    return result

def accuracy_score(y_values, predictions, tolerance = 1):
    result = list()
    t_predictions = tolerance_prediction(y_values, predictions, tolerance)
    conf = metrics.confusion_matrix(y_values, t_predictions)
    for row, _i in zip(conf, range(conf.shape[0])):
        for _j in range(conf.shape[0]):
            result.append(row[_i] / np.sum(row))

    return np.mean(result)

### Taken from scikit-learn.org confusion matrix example
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def get_sample_weights(target):
    weight = np.absolute(target - 6) + 1
    return np.power(weight * 3, 2)

def showData(data, indent=0):
    """Displays structure of data."""
    print()
    if isinstance(data, dict):
        for key in data:
            for _i in range(indent):
                print("\t", end="")
            print(key + " ", end="")
            showData(data[key], indent=indent + 1)

if __name__ == '__main__':
    main()

    # yval = [3,4,5,6,7,8,9,3,3,3,3,3,3,3]
    # ypred = [3,4,5,6,7,8,9,3,4,5,6,7,8,9]

    # print(accuracy_score(yval, ypred, 1.1))