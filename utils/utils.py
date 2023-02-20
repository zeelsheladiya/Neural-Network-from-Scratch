import numpy as np
import pandas as pd
import ast
from hstest import CheckResult


def one_hot(data):
    y_train = np.zeros((data.size, data.max() + 1))
    rows = np.arange(data.size)
    y_train[rows, data] = 1
    return y_train


def custom_uniform(min_value, max_value, shape, seed=3042022):
    rnd = np.random.default_rng(seed)
    return rnd.uniform(min_value, max_value, shape)


def get_list(s):
    index_from = s.find('[')
    index_to = s.find(']')
    data_str = s[index_from: index_to + 1]
    data_list = ast.literal_eval(data_str)
    if index_to + 2 > len(s):
        return data_list, None
    else:
        return data_list, s[index_to + 2:]


def full_check(result: list, true_result: list, name: str, tolerance=0.05):
    # Checking if the student's reply is a list
    if not isinstance(result, list):
        return CheckResult.wrong(f'Output for {name} is not a list.')

    # Checking size of the student's reply
    if len(result) != len(true_result):
        return CheckResult.wrong(f'Output for {name} should contain {len(true_result)} values,\
        found {len(result)}.')

    # Checking values of the student's reply
    for value, true_value in zip(result, true_result):
        if not (abs((value - true_value) / true_value) < tolerance):
            return CheckResult.wrong(f'Incorrect {name} values. Check your {name} function.')

    return None


def get_train():
    raw_train = pd.read_csv('../Data/fashion-mnist_train.csv')
    X_train = raw_train[raw_train.columns[1:]].values
    X_train = X_train / X_train.max()
    y_train = one_hot(raw_train['label'].values)
    return X_train, y_train


def get_test():
    raw_test = pd.read_csv('../Data/fashion-mnist_test.csv')
    X_test = raw_test[raw_test.columns[1:]].values
    X_test = X_test / X_test.max()  # Here should be used X_train.max() but in the dataset they are the same.
    y_test = one_hot(raw_test['label'].values)
    return X_test, y_test
