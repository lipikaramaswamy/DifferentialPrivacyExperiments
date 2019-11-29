# package imports
from tqdm import tqdm
import pandas as pd
import numpy as np

class LogisticRegression():
    def __init__(self, lr=0.01, num_iter=100000, fit_intercept=True, verbose=False):

        self.lr = lr
        self.num_iter = num_iter
        self.fit_intercept = fit_intercept
        self.verbose = verbose
        self.loss_by_epoch = None

    def __add_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)

    def __sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def __loss(self, h, y):
        '''
        Loss for logistic regression - negative log likelihood as we're trying to maximize likelihood
        '''
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()

    def fit(self, X, y, batch_size, sample_with_replacement = True):

        # list to cache loss by epoch - rewritten each time the model is fit, so should be accessed after each model run
        self.loss_by_epoch = []

        if self.fit_intercept:
            X = self.__add_intercept(X)

        # weights initializations
        self.theta = np.zeros(X.shape[1])

        for i in tqdm(range(self.num_iter)):
            # run through a random sample of training samples in each iteration
            rand_indices = np.random.choice(X.shape[0], size = batch_size, replace = sample_with_replacement)
            X_ = X[rand_indices, :]
            y_ = y[rand_indices]

            # calculate gradient
            z = np.dot(X_, self.theta)
            h = self.__sigmoid(z)
            gradient = np.dot(X_.T, (h - y_)) / y.size
            # update parameters
            self.theta -= self.lr * gradient

            #store loss
            z = np.dot(X, self.theta)
            h = self.__sigmoid(z)
            self.loss_by_epoch.append(self.__loss(h, y))

            if self.verbose == True and i % 5000 == 0:
                print(f'loss: {self.__loss(h, y)} \t')

    def predict_prob(self, X):
        if self.fit_intercept:
            X = self.__add_intercept(X)

        return self.__sigmoid(np.dot(X, self.theta))

    def predict(self, X, threshold):
        return self.predict_prob(X) >= threshold


class DPLogisticRegression():
    def __init__(self, lr=0.01, num_iter=100000,
                 fit_intercept=True, verbose=False,
                 clipping_param = 5, sigma = 2,
                 delta = 1e-6):

        self.lr = lr
        self.num_iter = num_iter
        self.fit_intercept = fit_intercept
        self.verbose = verbose
        self.loss_by_epoch = None

        # Privacy parameters
        self.clipping_param = clipping_param
        self.sigma = sigma
        self.delta = delta
        self.epsilon = None

    def __add_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)

    def __sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def __loss(self, h, y):
        '''
        Loss for logistic regression - negative log likelihood as we're trying to maximize likelihood
        '''
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()

    def __max_epsilon_moments_accountant(self, batch_size, X):
        '''
        Returns the maximum privacy loss (epsilon) using moments accountant
        (Theorem 1 from https://arxiv.org/pdf/1607.00133.pdf)
        '''
        return self.clipping_param/self.sigma*batch_size/X.shape[0]*np.sqrt(X.shape[0]//batch_size * np.log(1/self.delta))

    def fit(self, X, y, batch_size, repeat_data = True, sample_with_replacement = True):
        '''
        Fits the logistic regression model

        Inputs:
        - predictors, X
        - target, y
        - batch_size
        - repeat_data flag - True if each iteration can use each data point multiple times in different batches
        - sample_with_replacement flag - True if random batch sampling in SGD should be with replacement
        '''

        # list to cache loss by epoch - rewritten each time the model is fit, so should be accessed after each model run
        self.loss_by_epoch = []

        # fit intercept
        if self.fit_intercept:
            X = self.__add_intercept(X)

        # weights initialization
        self.theta = np.zeros(X.shape[1])

        # if only one pass over data is allowed in training
        if repeat_data == True:
            n_iter = self.num_iter
        else:
            n_iter = X.shape[0]//batch_size
            indices_shuffled = np.random.shuffle(np.arange(X.shape[0]))
            X_shuff = X[indices_shuffled,:].reshape(X.shape)
            y_shuff = y[indices_shuffled].reshape(-1,1)
#             print('X_shuff.shape',X_shuff.shape)
#             print('X.shape', X.shape)

        for i in tqdm(range(n_iter)):
            # fix this to not sample same row twice - or make an option
            if repeat_data == True:
                rand_indices = np.random.choice(X.shape[0], size = batch_size, replace = sample_with_replacement)
                X_ = X[rand_indices, :]
                y_ = y[rand_indices]

            if repeat_data == False:
                X_ = X_shuff[i*batch_size:(i+1)*batch_size, :]
                y_ = y_shuff[i*batch_size:(i+1)*batch_size]

            gradient_clipped = np.zeros(X_.shape)

            for row in range(X_.shape[0]):
                z_i = np.dot(X_[row,:], self.theta)
                h_i = self.__sigmoid(z_i)
                gradient_i = np.dot(X_[row,:].reshape(-1,X.shape[1]).T, (h_i - y_[row])) # / y_.size

                # clip the gradient using microbatch concept - fix this by calculating gradient example by example
                gradient_clipped_i = gradient_i/max(1, np.linalg.norm(gradient_i))
                gradient_clipped[row, :] = gradient_clipped_i.reshape(1,-1)

            gradient_noisy = 1/y_.size * ( np.sum(gradient_clipped, axis = 0) + np.random.normal(loc = 0.0, scale = self.sigma * self.clipping_param))

            self.theta -= self.lr * gradient_noisy

            #store loss
            z = np.dot(X, self.theta)
            h = self.__sigmoid(z)
            self.loss_by_epoch.append(self.__loss(h, y))

            if self.verbose == True and i % 1000 == 0:
                print(f'loss: {self.__loss(h, y)} \t')

        self.epsilon = self.__max_epsilon_moments_accountant(batch_size, X)

    def predict_prob(self, X):
        if self.fit_intercept:
            X = self.__add_intercept(X)

        return self.__sigmoid(np.dot(X, self.theta))

    def predict(self, X, threshold):
        return self.predict_prob(X) >= threshold
