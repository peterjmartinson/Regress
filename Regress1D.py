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

    def setNumberOfTrainingExamples(self, numpy_array):
        self.m = len(numpy_array)

    def incrementNumberOfFeatures(self):
        if self.n is None:
            self.n = 1
        else:
            self.n += 1

    def addAnotherTheta(self):
        if self.theta is None:
            self.theta = Coefficients(self.X.getNumberOfFeatures())
        else:
            self.theta = self.theta.addCoefficient()

    def setFeatures(self, numpy_array):
        if not isinstance(numpy_array, np.ndarray):
            raise TypeError('input must be a Numpy array')
        if self.X is None:
            self.X = TrainingInputs(numpy_array)
            self.m = self.X.getNumberOfTrainingExamples()
            self.n = self.X.getNumberOfFeatures()
        else:
            if len(numpy_array) == self.n:
                self.X.addTrainingExample(numpy_array)
            else:
                raise TypeError('input array is not the right size')
        self.m = self.X.getNumberOfTrainingExamples()
        self.addAnotherTheta()

    def getFeatures(self):
        return self.X

    def setTargets(self, numpy_array):
        if not isinstance(numpy_array, np.ndarray):
            raise TypeError('input must be a Numpy array')
        if numpy_array.size != self.m:
            raise ValueError('input has wrong number of training values')
        self.y = numpy_array

    def getTarget(self):
        return self.y

    # H(xi) = theta_0 + theta_1 * xi
    def getHypothesis(self):
        h = self.theta.getCoefficients()[0]*1 + np.sum(self.theta.getCoefficients()[1:] * self.X.getTrainingInputs(), axis=1)
        return h

    # J(theta_0, theta_1) = 1/2m sum_i=1^m [ (H(xi)-yi)^2 ]
    def getJ(self):
        h = self.getHypothesis()
        y = self.y
        m = self.m
        J = 0
        for i in range(len(y)):
            J = J + ( (h[i] - y[i])*(h[i] - y[i]) )
        J = J * 1/(2 * m)
        print("J = ", J)
        return J

    ## Note: the following is *only* valid for one-dimensional hypotheses,
    ## i.e. those with only two coefficients
    def getDerivativeOfJ(self):
        if self.X is None:
            raise ValueError('X contains no data!')
        if self.y is None:
            raise ValueError('y contains no data!')
        if self.m is None:
            raise ValueError('m contains no data!')
        if self.m == 0:
            raise ValueError('m is zero!  Division by Zero!')
        dJ = np.array([
            (1/self.m) * np.sum( (self.getHypothesis() - self.y) ),
            (1/self.m) * np.sum( (self.getHypothesis() - self.y) * self.X.getTrainingInputs().T)
        ])
        return dJ

    # theta_j := theta_j - alpha - [d/dtheta_j]J(theta_0, theta_1), for j = 0, 1
    def evaluateNewThetas(self):
        if self.theta is None:
            raise ValueError('theta contains no data!')
        if self.a is None:
            raise ValueError('a contains no data!')
        temp_theta = self.theta.getCoefficients() - self.a - self.getDerivativeOfJ()
        for i in range(len(temp_theta)):
            self.theta.updateCoefficient(i, temp_theta[i])

class ResidualSumOfSquares:
    """This is what Ng calls the 'Cost Function', or J"""

    def __init__(self):
        pass

    def getValue(self, hypothesis, targets):
        h = hypothesis
        y = targets
        m = len(y)
        J = 0
        for i in range(len(y)):
            J = J + ( (h[i] - y[i])*(h[i] - y[i]) )
        J = J * 1/(2 * m)
        print("J = ", J)
        return J

    def getDerivative(self):
        if self.X is None:
            raise ValueError('X contains no data!')
        if self.y is None:
            raise ValueError('y contains no data!')
        if self.m is None:
            raise ValueError('m contains no data!')
        if self.m == 0:
            raise ValueError('m is zero!  Division by Zero!')
        dJ = np.array([
            (1/self.m) * np.sum( (self.getHypothesis() - self.y) ),
            (1/self.m) * np.sum( (self.getHypothesis() - self.y) * self.X[1:])
        ])
        return dJ

class TrainingInputs:
    """This is the array of training inputs"""

    training_inputs = None
    number_of_features = 0
    number_of_training_examples = None

    # def setFeatures(self, numpy_array):
    #     if not isinstance(numpy_array, np.ndarray):
    #         raise TypeError('input must be a Numpy array')
    #     if self.X is None:
    #         self.setNumberOfTrainingExamples(numpy_array)
    #         self.X = np.vstack((np.ones((1, self.m)), numpy_array))
    #     else:
    #         if len(numpy_array) == self.m:
    #             self.X = np.vstack((self.X, numpy_array))
    #         else:
    #             raise TypeError('input array is not the right size')
    #     self.incrementNumberOfFeatures()
    #     self.addAnotherTheta()
    # Should retain information about number of features (n)
    # Should retain information about number of training examples (m)
    # Should allow adding training examples
    # Should *not* initialize the coefficients.  The Model class does this
    # should only know about the training inputs
    # set, get, add
    # Make "Hypothesis" do the work of prepending the x_0 = 1 term

    def __init__(self, numpy_array):
        if not isinstance(numpy_array, np.ndarray):
            raise TypeError('input must be a Numpy array')
        self.number_of_training_examples = len(numpy_array)
        self.training_inputs = numpy_array
        self.number_of_features += 1

    def getTrainingInputs(self):
        return self.training_inputs

    def getNumberOfTrainingExamples(self):
        return self.number_of_training_examples

    def getNumberOfFeatures(self):
        return self.number_of_features

    def addTrainingExample(self, numpy_array):
        if not isinstance(numpy_array, np.ndarray):
            raise TypeError('input must be a Numpy array')
        if len(numpy_array) != self.number_of_features:
            raise ValueError(f'input must be of size {self.number_of_features}!')
        self.training_inputs = np.vstack((self.training_inputs, numpy_array))
        self.number_of_training_examples += 1
        return self.training_inputs

    
# class TrainingTargets:

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
