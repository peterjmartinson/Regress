import pandas as pd
import numpy as np


class Model:
    """Provides a linear regression model"""
    
    def __init__(self):
        self.X     = None # Training input set
        self.y     = None # Target array
        self.m     = None # Number of training examples
        self.n     = None # Number of features
        self.beta  = None # Parameter array
        self.a     = 0.01  # Learning rate
        self.rss   = ResidualSumOfSquares()

    def addAnotherBeta(self):
        if self.beta is None:
            self.beta = Coefficients(self.X.getNumberOfFeatures())
        else:
            self.beta = self.beta.addCoefficient()

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
        self.addAnotherBeta()

    def setTargets(self, numpy_array):
        if not isinstance(numpy_array, np.ndarray):
            raise TypeError('input must be a Numpy array')
        if numpy_array.size != self.m:
            raise ValueError('input has wrong number of training values')
        self.y = TrainingResponses(numpy_array)

    # H(xi) = beta_0 + beta_1 * xi
    def getHypothesis(self):
        h = self.beta.getCoefficients()[0]*1 + np.sum(self.beta.getCoefficients()[1:] * self.X.getTrainingPredictors(), axis=1)
        return h

    # beta_j := beta_j - alpha - [d/dbeta_j]J(beta_0, beta_1), for j = 0, 1
    def evaluateNewBetas(self):
        if self.beta is None:
            raise ValueError('beta contains no data!')
        if self.a is None:
            raise ValueError('a contains no data!')
        dJ = self.rss.getDerivative(self.X.getTrainingPredictors(), self.y.getTrainingResponses(), self.getHypothesis())
        temp_beta = self.beta.getCoefficients() - self.a - dJ
        for i in range(len(temp_beta)):
            self.beta.updateCoefficient(i, temp_beta[i])

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
    """This is what Ng calls Theta, and ISL calls Beta.  The coefficients of the Hypothesis."""

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









### ===============  Test this thing out!
### ===============  Currently, the model wildly diverges.  Either there's something wrong with the model, or something wrong with the initial Coefficient guesses
### ===============  The "answer" should ideally be y = 1.7 * x, or beta = [0, 1.7]



model = Model()
training_predictors = np.array([[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]])
training_responses = np.array([1,2,3,4,5,6,7,8,9,10]) * 1.7
model.setPredictors(training_predictors)
model.setTargets(training_responses)

print(f'X:  {model.X.getTrainingPredictors()}')
print(f'y:  {model.y.getTrainingResponses()}')
print(f'm:  {model.m}')
print(f'n:  {model.n}')
print(f'beta:  {model.beta.getCoefficients()}')

model.evaluateNewBetas()
print(f'beta:  {model.beta.getCoefficients()}')
model.evaluateNewBetas()
print(f'beta:  {model.beta.getCoefficients()}')
model.evaluateNewBetas()
print(f'beta:  {model.beta.getCoefficients()}')
model.evaluateNewBetas()
print(f'beta:  {model.beta.getCoefficients()}')
model.evaluateNewBetas()
print(f'beta:  {model.beta.getCoefficients()}')
model.evaluateNewBetas()
print(f'beta:  {model.beta.getCoefficients()}')
model.evaluateNewBetas()
print(f'beta:  {model.beta.getCoefficients()}')
model.evaluateNewBetas()
print(f'beta:  {model.beta.getCoefficients()}')
model.evaluateNewBetas()
print(f'beta:  {model.beta.getCoefficients()}')
model.evaluateNewBetas()
print(f'beta:  {model.beta.getCoefficients()}')
model.evaluateNewBetas()
print(f'beta:  {model.beta.getCoefficients()}')
model.evaluateNewBetas()
print(f'beta:  {model.beta.getCoefficients()}')
model.evaluateNewBetas()
print(f'beta:  {model.beta.getCoefficients()}')

print("--------------------")

## Note, beta_0 = 0, beta_1 = 1.7
