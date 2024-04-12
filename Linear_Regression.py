import numpy as np
import random
import pandas as pd


class MyLineReg:
    def __init__(self, n_iter=100, learning_rate=0.1, weights=None, metric=None, reg=None, l1_coef=0, l2_coef=0,
                 sgd_sample=None, random_state=42):
        self.n_iter, self.learning_rate = n_iter, learning_rate
        self.weights, self.metric = weights, metric
        self.sgd_sample, self.random_state = sgd_sample, random_state
        self.X, self.y = 0, 0
        self.reg = reg
        self.l1_coef, self.l2_coef = l1_coef, l2_coef
        if isinstance(self.learning_rate, float):
            self.flag = 0
        else:
            self.f = self.learning_rate
            self.flag = 1

    def __str__(self):
        return self.weights.mean()

    def fit(self, X, y, verbose=False):
        random.seed(self.random_state)
        array_ones = np.ones((X.shape[0], 1))
        X_for_d = X.copy()
        X = np.hstack([array_ones, X])
        self.weights = np.array([1] * (X.shape[1]))
        for i in range(1, self.n_iter + 1):
            if self.flag:
                self.learning_rate = self.f(i)
            if self.sgd_sample:
                if isinstance(self.sgd_sample, float):
                    index_X = random.sample(range(X_for_d.shape[0]), int(np.round(self.sgd_sample * X_for_d.shape[0])))
                    X_new = X_for_d.iloc[index_X]
                    y_new = y.iloc[index_X]
                else:
                    index_X = random.sample(range(X_for_d.shape[0]), self.sgd_sample)
                    X_new = X_for_d.iloc[index_X]
                    y_new = y.iloc[index_X]
                array_ones = np.ones((X_new.shape[0], 1))
                X_new = np.hstack([array_ones, X_new])
                y_predict = X_new @ self.weights
                # print(y_predict.shape)
                # print(y.shape)
                self.gradient_descent(X_new, y_predict, y_new)
            else:
                y_predict = X @ self.weights
                self.gradient_descent(X, y_predict, y)
        self.X = X
        self.y = y

    def gradient_descent(self, X, y_predict, y):
        if self.reg == 'l1':
            grad = (2 * ((y_predict - y) @ X)) / X.shape[0] + self.l1_coef * np.sign(self.weights)
            self.weights = self.weights - self.learning_rate * grad
        elif self.reg == 'l2':
            grad = (2 * ((y_predict - y) @ X)) / X.shape[0] + self.l2_coef * 2 * self.weights
            self.weights = self.weights - self.learning_rate * grad
        elif self.reg == 'elasticnet':
            grad = (2 * ((y_predict - y) @ X)) / X.shape[0] + self.l1_coef * np.sign(
                self.weights) + self.l2_coef * 2 * self.weights
            self.weights = self.weights - self.learning_rate * grad
        else:
            grad = (2 * ((y_predict - y) @ X)) / X.shape[0]
            self.weights = self.weights - self.learning_rate * grad

    def predict(self, X):
        array_ones = np.ones((X.shape[0], 1))
        X = np.hstack([array_ones, X])
        y_predict = X @ self.weights
        return y_predict

    def get_coef(self):
        return self.weights[1:]

    def get_best_score(self):
        predict = self.X @ self.weights
        if self.metric == 'mae':
            return self.mean_absolute_error(self.y, predict)
        elif self.metric == 'mse':
            return self.mean_squared_error(self.y, predict)
        elif self.metric == 'rmse':
            return self.root_mean_squared_error(self.y, predict)
        elif self.metric == 'r2':
            return self.R2_score(self.y, predict)
        else:
            return self.mean_absolute_percentage_error(self.y, predict)

    def mean_squared_error(self, y_true, predict):
        MSE = (((y_true - predict) ** 2).sum()) / len(y_true)
        return MSE

    def root_mean_squared_error(self, y_true, predict):
        RMSE = np.sqrt((((y_true - predict) ** 2).sum()) / len(y_true))
        return RMSE

    def mean_absolute_error(self, y_true, predict):
        MAE = (abs(y_true - predict).sum()) / len(y_true)
        return MAE

    def R2_score(self, y_true, predict):
        R2 = 1 - ((((y_true - predict) ** 2).sum()) / len(y_true)) / (
                (((y_true - y_true.mean()) ** 2).sum()) / len(y_true))
        return R2

    def mean_absolute_percentage_error(self, y_true, predict):
        MAPE = (100 * abs((y_true - predict) / y_true).sum()) / len(y_true)
        return MAPE
