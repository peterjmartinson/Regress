import pandas as pd
import numpy as np

# class Regress1D(object):
class Regress1D:
    """Provides a linear regression model"""
    
    def __init__(self):
        self.X     = None # Feature array
        self.y     = None # Target array
        self.m     = None # Number of training examples
        self.n     = None # Number of features
        self.theta = None # Parameter array
        self.dJ    = None # container for all current dJ/dTheta values
        self.a     = 0.1  # Learning rate

    def setNumberOfTrainingExamples(self, numpy_array):
        self.m = len(numpy_array)

    def incrementNumberOfFeatures(self):
        if self.n is None:
            self.n = 1
        else:
            self.n += 1

    def addAnotherTheta(self):
        if self.theta is None:
            self.theta = np.array([1, 1])
        else:
            self.theta = np.append(self.theta, [1])

    def setFeatures(self, numpy_array):
        if not isinstance(numpy_array, np.ndarray):
            raise TypeError('input must be a Numpy array')
        if self.X is None:
            self.setNumberOfTrainingExamples(numpy_array)
            self.X = np.vstack((np.ones((1, self.m)), numpy_array))
        else:
            if len(numpy_array) == self.m:
                self.X = np.vstack((self.X, numpy_array))
            else:
                raise TypeError('input array is not the right size')
        self.incrementNumberOfFeatures()
        self.addAnotherTheta()

    def getFeatures(self):
        return self.X

    def setTargets(self, numpy_array):
        if not isinstance(numpy_array, np.ndarray):
            raise TypeError('input must be a Numpy array')
        self.y = numpy_array

    def getTarget(self):
        return self.y

    # np.sum(np.sum(theta*X.T, axis=1) - y)
    # where X = [[1,1,1...],[x0_0, x0_1, x0_2, ...],[x1_0, x1_1, x1_2, ...],...]
    def dJ_over_dTheta1(self):
        result = 0
        for i in range(0, self.m):
            result += self.theta[0] + self.theta[1]*self.X[i] - self.y[i]
        return result/self.m

    def dJ_over_dTheta2(self):
        result = 0
        for i in range(0, self.m):
            result += (self.theta[0] + self.theta[1]*self.X[i] - self.y[i]) * self.X[i]
        return result/self.m

    def evaluateDerivativeOfJ(self):
        if self.X is None:
            raise ValueError('X contains no data!')
        if self.y is None:
            raise ValueError('y contains no data!')
        if self.m is None:
            raise ValueError('m contains no data!')
        if self.m == 0:
            raise ValueError('m is zero!  Division by Zero!')
        self.dJ = np.array([
            (1/self.m) * np.sum( (np.sum(self.theta*self.X.T, axis=1) - self.y) ),
            (1/self.m) * np.sum( (np.sum(self.theta*self.X.T, axis=1) - self.y) * self.X[1:])
        ])

    # theta_j := theta_j - alpha - [d/dtheta_j]J(theta_0, theta_1), for j = 0, 1
    def evaluateNewThetas(self):
        if self.theta is None:
            raise ValueError('theta contains no data!')
        if self.a is None:
            raise ValueError('a contains no data!')
        self.evaluateDerivativeOfJ()
        if self.dJ is None:
            raise ValueError('dJ contains no data!')
        # temp_theta = np.zeros(len(self.theta))
        temp_theta = self.theta - self.a - self.dJ
        self.theta = temp_theta
        # for j in range(len(self.theta)):
        #     temp_theta[j] = self.theta[j] - self.a - self.dJ
        ## self.theta[0] = 2

