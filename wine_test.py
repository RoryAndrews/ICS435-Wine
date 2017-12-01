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
    scaler = MaxAbsScaler()

    # Process red wine data
    red_data = pd.read_csv("winequality-red.csv")
    features, target = process_wine_data(red_data)
    params = {'C': 5.3366992312063068, 'epsilon': 0.18181818181818182, 'gamma': 0.01}
    data["red-unscaled"] = (features, target, params)
    scaler.fit(features)
    features = scaler.transform(features) # Scale features to be similar
    params = {'C': 0.65793322465756787, 'epsilon': 0.0, 'gamma': 5.3366992312063068}
    data["red-max-abs-scaled"] = (features, target, params)

    # Process white wine data
    white_data = pd.read_csv("winequality-white.csv", sep=';')
    features, target = process_wine_data(white_data)
    params = {'C': 0.01, 'epsilon': 2.0, 'gamma': 0.01}
    data["white-unscaled"] = (features, target, params)
    scaler.fit(features)
    features = scaler.transform(features) # Scale features to be similar
    params = {'C': 0.65793322465756787, 'epsilon': 0.90909090909090917, 'gamma': 5.3366992312063068}
    data["white-max-abs-scaled"] = (features, target, params)

    # Combine and process both wine data
    combined_data = pd.concat([red_data, white_data])
    combined_data = shuffle(combined_data)
    features, target = process_wine_data(combined_data)
    params = {'C': 1.873817422860383, 'epsilon': 0.0, 'gamma': 0.23101297000831592}
    data["combined-unscaled"] = (features, target, params)
    scaler.fit(features)
    features = scaler.transform(features) # Scale features to be similar
    params = {'C': 1.873817422860383, 'epsilon': 0.0, 'gamma': 123.28467394420659}
    data["combined-max-abs-scaled"] = (features, target, params)

    # for key in data:
    #     print(key)
    #     print(data[key][0])
    #     print(data[key][1])
    #     print(data[key][2])

    parameter_test(data)

    # quality_prediction(data)

    # features, target = load_wine(return_X_y=True)
    # scaler = StandardScaler()
    # scaler.fit(features)

    # data = scaler.transform(features)

    # x_train, x_test, y_train, y_test = train_test_split(data, target,
    #                                                     test_size=0.30)

    # clf = NuSVC()

    # clf.fit(x_train, y_train)
    # prediction = clf.predict(x_test)

    # print(metrics.accuracy_score(y_test, prediction))

def parameter_test(data):
    """Test."""
    for key in data:
        features = data[key][0]
        target = data[key][1]
        print(key)
        print(best_SVR_parameters2((features, target)))
        # print(best_SVR_parameters((features, target)))

def quality_prediction(data):
    """Outputs the mean absolute error and the confusion matrix"""    
    for key in data:
        confusion_results = list()
        mae_results = list()
        features = data[key][0]
        target = data[key][1]
        params = data[key][2]
        estimator = SVR(**params, kernel='rbf')
        for _i in range(50):
            x_train, x_test, y_train, y_test = train_test_split(features, target,
                                                                test_size=0.33)
            estimator.fit(x_train, y_train)
            prediction = estimator.predict(x_test)

            mae = metrics.mean_absolute_error(y_test, prediction)
            round_pred = [int(i) for i in np.around(prediction)]
            conf = metrics.confusion_matrix(y_test, round_pred, labels=range(3,10))
            # print()
            # print(key)
            # print(mae)
            # print(conf)
            confusion_results.append(conf)
            mae_results.append(mae)
        print(key)
        print("MAE mean: %f" % np.mean(mae_results))
        np.set_printoptions(precision=2, suppress=True)
        avg_conf = average_confusion(confusion_results)
        plt.clf()
        plot_confusion_matrix(avg_conf, range(3,10),
                              title=key + '\nmean absolute error = ' + str(np.mean(mae_results)))
        plt.savefig(key + "-cm", dpi=150)
        


def best_SVR_parameters(data):
    """Returns best parameters for SVR."""
    features = data[0]
    target = data[1]

    estimator = SVR(max_iter=1000, kernel='rbf')

    dynamic_C_range = np.logspace(-2, 3, 12)
    dynamic_gamma_range = np.logspace(-2, 3, 12)
    dynamic_epsilon_range = np.linspace(0, 2, 12)
    param_grid = dict(C=dynamic_C_range, epsilon=dynamic_epsilon_range, gamma=dynamic_gamma_range)

    grid = GridSearchCV(estimator, param_grid=param_grid, n_jobs=4)
    start = time.time()
    grid.fit(features, target)
    print("Time: " + str(time.time() - start))
    return grid.best_params_


def best_SVR_parameters2(data):
    """Returns best parameters for SVR."""
    features = data[0]
    target = data[1]


    estimator = SVR(max_iter=1000, kernel='rbf')

    dynamic_C_range = np.logspace(-10, 8, 12, base=2)
    dynamic_gamma_range = np.logspace(-10, 8, 12, base=2)
    dynamic_epsilon_range = np.linspace(0, 1, 12)
    param_grid = dict(C=dynamic_C_range, epsilon=dynamic_epsilon_range, gamma=dynamic_gamma_range)
    weights = get_sample_weights(target)

    grid = GridSearchCV(estimator, param_grid=param_grid, n_jobs=4, scoring='neg_mean_absolute_error')
    start = time.time()
    grid.fit(features, target, sample_weight=weights)
    print("Time: " + str(time.time() - start))
    return grid.best_params_

def average_confusion(confusion_results):
    """confusion"""
    mean = np.zeros_like(confusion_results[0])

    for conf in confusion_results:
        mean += conf

    mean = mean / len(confusion_results)
    return mean

def process_wine_data(data):
    """Returns features and target of wine data."""
    features = data.loc[:, "fixed acidity":"alcohol"].as_matrix()
    target = data.loc[:, "quality"].as_matrix()
    return features, target

def get_sample_weights(target):
    weight = np.absolute(target - 6) + 1
    return np.power(weight, 2)

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

    fmt = '.1f' # if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

if __name__ == '__main__':
    main()
