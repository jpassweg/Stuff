import numpy as np
from sklearn.metrics import mean_squared_error
import math


def kfold_split(a, div_index, divisions, permutation = [-1]):
    a_train = []
    a_validate = []
    length = float(a.shape[0])
    if(permutation[0] == -1): permutation = np.arange(length)
    multiplier = int(length / divisions)
    for i in range(multiplier * div_index):
        a_train.append(a[permutation[i]])
    for i in range(multiplier * div_index, multiplier * (div_index+1)):
        a_validate.append(a[permutation[i]])
    for i in range(multiplier * (div_index+1), int(length)):
        a_train.append(a[permutation[i]])
    return np.array(a_train), np.array(a_validate)


def cross_validation(x, y, permutation, k_fold, function, lamda = 1):
    result = 0
    for j in range(k_fold):
        x_train, x_validate = kfold_split(x, j, k_fold, permutation)
        y_train, y_validate = kfold_split(y, j, k_fold, permutation)
        weights = function(x_train, y_train, lamda)
        error = rmse(weights, x_validate, y_validate)
        result += error / float(k_fold)


def closed_form_ridge_regression(x, y, lamda = 1):
    length = x.shape[1]
    x_transpose = np.transpose(x)
    quad = np.add(np.dot(x_transpose, x),np.identity(length) * lamda)
    quad_inv = np.linalg.inv(quad)
    return np.dot(np.dot(quad_inv, x_transpose), y)


def closed_form_linear_regression(x, y, lamda = 1):
    x_transpose = np.transpose(x)
    quad = np.dot(x_transpose, x)
    quad_inv = np.linalg.inv(quad)
    return np.dot(np.dot(quad_inv, x_transpose), y)


def gradient_descent(x_train, y_train, function, epsilon = 0, learn_rate = 000000000.1, lamda = 1):
    error = float("inf")
    prev_error = 0
    m = np.arange(x_train.shape[1])
    counter = 1
    while math.fabs(error - prev_error) > epsilon:
        m = function(x_train, y_train, m, learn_rate, lamda)
        prev_error = error
        error = rmse(m, x_train, y_train)
        if(counter % 2000 == 0):
            print("error after %10d iterations is %8.18f" % (counter, error))
        if(error > prev_error):
            print("new error bigger, maybe something wrong")
            break
        counter = counter + 1
    return m


def __gradient_descent_iteration_intern(x_train, y_train, m, learn_rate):
    y_pred = x_train.dot(m)
    residual = np.subtract(y_train, y_pred)
    x_transpose = np.transpose(x_train)
    d_m = x_transpose.dot(residual)
    d_m = np.multiply(d_m,-learn_rate)
    return d_m


def gradient_descent_iteration_least_square(x_train, y_train, m, learn_rate, lamda = 1):
    d_m = __gradient_descent_iteration_intern(x_train, y_train, m, learn_rate)
    m = np.subtract(m,d_m)
    return m


def gradient_descent_iteration_ridge(x_train, y_train, m, learn_rate, lamda = 1):
    d_m = __gradient_descent_iteration_intern(x_train, y_train, m, learn_rate)
    if(2*learn_rate*lamda < 1):
        m = np.subtract((1-2*learn_rate*lamda)*m,d_m)
    return m


def gradient_descent_iteration_lasso(x_train, y_train, m, learn_rate, lamda = 1):
    d_m = __gradient_descent_iteration_intern(x_train, y_train, m, learn_rate)
    if(learn_rate*lamda < 1):
        m = np.subtract((1-learn_rate*lamda)*m,d_m)
    return m


def rmse(w, x, y):
    return mean_squared_error(y, np.dot(x,w))**0.5 
