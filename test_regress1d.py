import unittest
import numpy as np
import numpy.testing as npt
from Regress1D import Regress1D

class TestCanary(unittest.TestCase):
    
    def test_Tweets(self):
        assert 1 == 1

class Test_setFeatures(unittest.TestCase):

    def setUp(self):
        self.model = Regress1D()
        self.numpy_array = np.array([1.,2.,3.])
        self.m = 3

    def test__Exists(self):
        self.assertTrue(hasattr(self.model, 'setFeatures'))

    def test__TakesNumpyArray(self):
        not_an_array = "not a Numpy ndarray"
        with self.assertRaises(TypeError):
            self.model.setFeatures(not_an_array)

    def test__Set_X_if_X_is_Empty(self):
        correct_X = np.array([[1,1,1],[1.,2.,3.]])
        self.model.setFeatures(self.numpy_array)
        npt.assert_array_equal(self.model.X, correct_X)

    def test__Set_m_if_X_is_Empty(self):
        self.model.setFeatures(self.numpy_array)
        assert self.m == self.model.m

    def test__Throw_Exception_if_Array_is_Wrong_Size(self):
        self.model.X = np.array([1.,1.,1.,1.])
        self.model.m = 4
        with self.assertRaises(TypeError):
            self.model.setFeatures(self.numpy_array)

    def test__Append_Array_if_X_is_Set(self):
        self.model.X = np.array([[0.1,0.1,0.1],[1.,2.,3.]])
        self.model.m = 3
        correct_X = np.array([[0.1,0.1,0.1],[1.,2.,3.],[1.,2.,3.]])
        self.model.setFeatures(self.numpy_array)
        npt.assert_array_equal(self.model.X, correct_X)

    def test__Set_n_equal_to_1_if_X_is_Empty(self):
        self.model.setFeatures(self.numpy_array)
        assert self.model.n == 1

    def test__Increment_n_if_X_is_Set(self):
        self.model.setFeatures(self.numpy_array)
        self.model.setFeatures(self.numpy_array)
        assert self.model.n == 2

    def test__Set_theta_if_theta_is_not_yet_set(self):
        correct_theta = np.array([1, 1])
        self.model.setFeatures(self.numpy_array)
        npt.assert_array_equal(self.model.theta, correct_theta)

    def test__Add_another_theta_if_theta_is_already_set(self):
        correct_theta = np.array([1,1,1,1])
        self.model.setFeatures(self.numpy_array)
        self.model.setFeatures(self.numpy_array)
        self.model.setFeatures(self.numpy_array)
        npt.assert_array_equal(self.model.theta, correct_theta)


class Test_setTargets(unittest.TestCase):

    def setUp(self):
        self.model = Regress1D()
        self.numpy_array = np.array([1,2,3])

    def test__Exists(self):
        self.assertTrue(hasattr(self.model, 'setTargets'))

    def test__TakesNumpyArray(self):
        not_an_array = "not a Numpy ndarray"
        with self.assertRaises(TypeError):
            self.model.setTargets(not_an_array)

    def test__Sets_y(self):
        self.model.setTargets(self.numpy_array)
        npt.assert_array_equal(self.model.y, self.numpy_array)


class Test_evaluateDerivativeOfJ(unittest.TestCase):

    def setUp(self):
        self.model = Regress1D()
        self.feature_array = np.array([1,2,3,4,5,6,7,8,9,10])
        self.target_array = self.feature_array * 1.7

    def test__Exists(self):
        self.assertTrue(hasattr(self.model, 'evaluateDerivativeOfJ'))

    def test__Throws_If_X_Is_None(self):
        with self.assertRaises(ValueError):
            self.model.evaluateDerivativeOfJ()

    def test__Throws_If_y_Is_None(self):
        self.model.setFeatures(self.feature_array)
        with self.assertRaises(ValueError):
            self.model.evaluateDerivativeOfJ()

    ## Need to figure out a test value for dJ, and the rest of the inputs
    def test__Sets_dJ(self):
        correct_dJ = [-2.85, -21.45]
        self.model.setFeatures(self.feature_array)
        self.model.setTargets(self.target_array)
        self.model.evaluateDerivativeOfJ()
        print("--- Diagnostic ---")
        print("m =     ", self.model.m)
        print("n =     ", self.model.n)
        print("X =\n", self.model.X)
        print("y =     ", self.model.y)
        print("theta = ", self.model.theta)
        print("dJ =    ", self.model.dJ)
        npt.assert_array_equal(self.model.dJ, correct_dJ)

class Test__evaluateNewThetas(unittest.TestCase):

    def setUp(self):
        self.model = Regress1D()
        self.feature_array = np.array([1,2,3,4,5,6,7,8,9,10])
        self.target_array = self.feature_array * 1.7

    def test__Exists(self):
        self.assertTrue(hasattr(self.model, 'evaluateNewThetas'))

    def test__Throws_If_dJ_Is_None(self):
        self.model.X = 1
        self.model.y = 1
        self.model.theta = 1
        self.model.dJ = None
        self.model.a = 1
        with self.assertRaises(ValueError):
            self.model.evaluateNewThetas()

    def test__Throws_If_theta_Is_None(self):
        self.model.X = 1
        self.model.y = 1
        self.model.theta = None
        self.model.dJ = 1
        self.model.a = 1
        with self.assertRaises(ValueError):
            self.model.evaluateNewThetas()

    def test__Throws_If_a_Is_None(self):
        self.model.X = 1
        self.model.y = 1
        self.model.theta = 1
        self.model.dJ = 1
        self.model.a = None
        with self.assertRaises(ValueError):
            self.model.evaluateNewThetas()


