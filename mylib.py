import numpy as np
from sklearn.metrics import mean_squared_error
import math

'''
tried to do a library for ml projects with basic functionality
however generalizing resulted in a lot of work adding complexity
to the methods which is difficult to handle for such a small project
'''


'''
methods used for cross validation
'''

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


def cross_validation(x, y, k_fold, training_function, 
        permutation = np.array([-1]), epsilon = 0, learn_rate = 000000000.1, lamdas = np.array([1]), error_function = None, gradient_version = None):
    if(permutation[0] == -1): permutation = np.random.permutation(x.shape[0])
    total_result = []
    if(error_function == None): error_function = rmse
    for i in range(len(lamdas)):
        result = 0
        for j in range(k_fold):
            x_train, x_validate = kfold_split(x, j, k_fold, permutation)
            y_train, y_validate = kfold_split(y, j, k_fold, permutation)
            weights = training_function(x_train, y_train, function=gradient_version, epsilon=epsilon, learn_rate=learn_rate, lamda=lamdas[i])
            error = error_function(weights, x_validate, y_validate)
            result += error / float(k_fold)
        total_result.append(result)
    return total_result

'''
different regressions
'''

def closed_form_ridge_regression(x, y, function = None, epsilon = 0, learn_rate = 000000000.1, lamda = 1):
    length = x.shape[1]
    x_transpose = np.transpose(x)
    quad = np.add(np.dot(x_transpose, x),np.identity(length) * lamda)
    quad_inv = np.linalg.inv(quad)
    return np.dot(np.dot(quad_inv, x_transpose), y)


def closed_form_linear_regression(x, y, function = None, epsilon = 0, learn_rate = 000000000.1, lamda = 1):
    x_transpose = np.transpose(x)
    quad = np.dot(x_transpose, x)
    quad_inv = np.linalg.inv(quad)
    return np.dot(np.dot(quad_inv, x_transpose), y)


# choose learn_rate = -1 to have an adaptive learn_rate starting with 0.1
def gradient_descent(x_train, y_train, function, epsilon = 0, learn_rate = 000000000.1, lamda = 1):
    error = float("inf")
    prev_error = 0
    m = np.arange(x_train.shape[1])
    counter = 1
    adaptive_learn_rate = 0
    if(learn_rate == -1):
        learn_rate = 0.1
        adaptive_learn_rate = 1

    while math.fabs(error - prev_error) > epsilon:
        m = function(x_train, y_train, m, learn_rate, lamda)
        prev_error = error
        error = rmse(m, x_train, y_train)
        if(counter % 2000 == 0):
            print("error after %10d iterations is %8.18f" % (counter, error))
            if(adaptive_learn_rate==1): 
                learn_rate = learn_rate * 2.0
        if(error > prev_error):
            if(adaptive_learn_rate==1):
                print("new error bigger, maybe something wrong")
                break
            else:
                learn_rate = learn_rate / 10.0
        counter = counter + 1
    return m

'''
different gradient descents
'''

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

'''
other methods
'''

# returns the square root of the mean of |y-w*x| squared
def rmse(w, x, y):
    return mean_squared_error(y, np.dot(x,w))**0.5 
