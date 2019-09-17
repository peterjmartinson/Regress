import pandas as pd
import numpy as np

class Model:
    """Provides a linear regression model"""
    
    def __init__(self):
        self.X     = None # Training input set
        self.y     = None # Target array
        self.m     = None # Number of training examples
        self.n     = None # Number of features
        self.theta = None # Parameter array
        self.a     = 0.1  # Learning rate
        self.rss   = ResidualSumOfSquares()


    def addAnotherTheta(self):
        if self.theta is None:
            self.theta = Coefficients(self.X.getNumberOfFeatures())
        else:
            self.theta = self.theta.addCoefficient()

    def setPredictors(self, numpy_array):
        if not isinstance(numpy_array, np.ndarray):
            raise TypeError('input must be a Numpy array')
        if self.X is None:
            self.X = TrainingPredictors(numpy_array)
            self.m = self.X.getNumberOfTrainingPredictors()
            self.n = self.X.getNumberOfFeatures()
        else:
            if len(numpy_array) == self.n:
                self.X.addTrainingPredictor(numpy_array)
            else:
                raise TypeError('input array is not the right size')
        self.m = self.X.getNumberOfTrainingPredictors()
        self.addAnotherTheta()


    def setTargets(self, numpy_array):
        if not isinstance(numpy_array, np.ndarray):
            raise TypeError('input must be a Numpy array')
        if numpy_array.size != self.m:
            raise ValueError('input has wrong number of training values')
        self.y = numpy_array


    # H(xi) = theta_0 + theta_1 * xi
    def getHypothesis(self):
        h = self.theta.getCoefficients()[0]*1 + np.sum(self.theta.getCoefficients()[1:] * self.X.getTrainingPredictors(), axis=1)
        return h

    # theta_j := theta_j - alpha - [d/dtheta_j]J(theta_0, theta_1), for j = 0, 1
    def evaluateNewThetas(self):
        if self.theta is None:
            raise ValueError('theta contains no data!')
        if self.a is None:
            raise ValueError('a contains no data!')
        dJ = self.rss.getDerivative(self.X.getTrainingPredictors(), self.y, self.getHypothesis())
        temp_theta = self.theta.getCoefficients() - self.a - dJ
        for i in range(len(temp_theta)):
            self.theta.updateCoefficient(i, temp_theta[i])

class ResidualSumOfSquares:
    """This is what Ng calls the 'Cost Function', or J"""

    def __init__(self):
        pass

    def getValue(self, targets, hypothesis):
        h = hypothesis
        y = targets
        m = len(y)
        J = 0
        for i in range(len(y)):
            J = J + ( (h[i] - y[i])*(h[i] - y[i]) )
        J = J * 1/(2 * m)
        print("J = ", J)
        return J

    def getDerivative(self, inputs, targets, hypothesis):
        if not isinstance(inputs, np.ndarray):
            raise TypeError('Input must be a Numpy array')
        if not isinstance(targets, np.ndarray):
            raise TypeError('Target must be a Numpy array')
        if not isinstance(hypothesis, np.ndarray):
            raise TypeError('Hypothesis must be a Numpy array')
        if len(targets) == 0:
            raise ZeroDivisionError('m is zero!  Division by Zero!')
        h = hypothesis
        x = inputs.T
        y = targets
        m = len(y)
        dJ = np.array([
            (1/m) * np.sum( (h - y) ),
            (1/m) * np.sum( (h - y) * x)
        ])
        return dJ

class TrainingPredictors:
    """This is the array of training inputs"""

    training_predictors = None
    number_of_features = 0
    number_of_training_predictors = None

    def __init__(self, numpy_array):
        if not isinstance(numpy_array, np.ndarray):
            raise TypeError('input must be a Numpy array')
        self.number_of_training_predictors = len(numpy_array)
        self.training_predictors = numpy_array
        self.number_of_features += 1

    def getTrainingPredictors(self):
        return self.training_predictors

    def getNumberOfTrainingPredictors(self):
        return self.number_of_training_predictors

    def getNumberOfFeatures(self):
        return self.number_of_features

    def addTrainingPredictor(self, numpy_array):
        if not isinstance(numpy_array, np.ndarray):
            raise TypeError('input must be a Numpy array')
        if len(numpy_array) != self.number_of_features:
            raise ValueError(f'input must be of size {self.number_of_features}!')
        self.training_predictors = np.vstack((self.training_predictors, numpy_array))
        self.number_of_training_predictors += 1
        return self.training_predictors

    
class TrainingResponses:
    """This is the array of 'y-values'"""

    training_responses = None

    def __init__(self, numpy_array):
        if not isinstance(numpy_array, np.ndarray):
            raise TypeError('input must be a Numpy array')
        self.training_responses = numpy_array

    def getTrainingResponses(self):
        return self.training_responses

    def addTrainingResponse(self, added_training_response):
        if not isinstance(added_training_response, np.ndarray):
            raise TypeError('input must be a Numpy array')
        self.training_responses = np.append(self.training_responses, added_training_response)
        return self.training_responses

class Coefficients:
    """This is what Ng calls Theta.  The coefficients of the Hypothesis."""

    c = None

    def __init__(self, number_of_features):
        if not isinstance(number_of_features, int):
            raise TypeError('input must be an integer')
        self.c = np.ones(number_of_features + 1)

    def getCoefficients(self):
        return self.c

    def addCoefficient(self):
        self.c = np.append(self.c, [1])
        return self.c

    def updateCoefficient(self, index, replacement_element):
        if index > len(self.c):
            raise ValueError('index out of range')
        if not isinstance(replacement_element, float):
            raise TypeError('Replacement element must by of type *float*')
        self.c[index] = replacement_element
        return self.c


# class Hypothesis:
#     """This is the result of running the model"""

# class Model:
#     """This is the glue"""
