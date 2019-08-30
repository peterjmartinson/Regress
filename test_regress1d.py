import pytest
import numpy as np
import numpy.testing as npt

def printDiagnostics(self):
    print("--- Diagnostic ---")
    print("X (Feature array):               ", self.X)
    print("y (Target array):                ", self.y)
    print("m (Number of training examples): ", self.m)
    print("n (Number of features):          ", self.n)
    print("theta (Parameter array):         ", self.theta)
    print("dJ (container for all current dJ/dTheta values): ", self.dJ)
    print("a (Learning rate):               ", self.a)

## ======================================================= Pytest Fixtures

@pytest.fixture
def model():
    from Regress1D import Regress1D
    return Regress1D()

@pytest.fixture
def numpy_array():
    return np.array([1., 2., 3.])

@pytest.fixture
def feature_array():
    return np.array([1,2,3,4,5,6,7,8,9,10])

@pytest.fixture
def target_array():
    feature_array = np.array([1,2,3,4,5,6,7,8,9,10])
    return feature_array * 1.7

@pytest.fixture
def m():
    return 3

## ======================================================= Tests

class TestCanary:
    
    def test_Tweets(self):
        assert 1 == 1

class Test_setFeatures:

    def test__Exists(self, model):
        assert hasattr(model, 'setFeatures')

    def test__TakesNumpyArray(self, model):
        not_an_array = "not a Numpy ndarray"
        with pytest.raises(TypeError):
            model.setFeatures(not_an_array)

    def test__Set_X_if_X_is_Empty(self, model, numpy_array):
        correct_X = np.array([[1,1,1],[1.,2.,3.]])
        model.setFeatures(numpy_array)
        npt.assert_array_equal(model.X, correct_X)

    def test__Set_m_if_X_is_Empty(self, model, numpy_array, m):
        model.setFeatures(numpy_array)
        assert m == model.m

    def test__Throw_Exception_if_Array_is_Wrong_Size(self, model, numpy_array):
        model.X = np.array([1.,1.,1.,1.])
        model.m = 4
        with pytest.raises(TypeError):
            model.setFeatures(numpy_array)

    def test__Append_Array_if_X_is_Set(self, model, numpy_array):
        model.X = np.array([[0.1,0.1,0.1],[1.,2.,3.]])
        model.m = 3
        correct_X = np.array([[0.1,0.1,0.1],[1.,2.,3.],[1.,2.,3.]])
        model.setFeatures(numpy_array)
        npt.assert_array_equal(model.X, correct_X)

    def test__Set_n_equal_to_1_if_X_is_Empty(self, model, numpy_array):
        model.setFeatures(numpy_array)
        assert model.n == 1

    def test__Increment_n_if_X_is_Set(self, model, numpy_array):
        model.setFeatures(numpy_array)
        model.setFeatures(numpy_array)
        assert model.n == 2

    def test__Set_theta_if_theta_is_not_yet_set(self, model, numpy_array):
        correct_theta = np.array([1, 1])
        model.setFeatures(numpy_array)
        npt.assert_array_equal(model.theta, correct_theta)

    def test__Add_another_theta_if_theta_is_already_set(self, model, numpy_array):
        correct_theta = np.array([1,1,1,1])
        model.setFeatures(numpy_array)
        model.setFeatures(numpy_array)
        model.setFeatures(numpy_array)
        npt.assert_array_equal(model.theta, correct_theta)


class Test_setTargets:

    def test__Exists(self, model):
        assert hasattr(model, 'setTargets')

    def test__TakesNumpyArray(self, model):
        not_an_array = "not a Numpy ndarray"
        with pytest.raises(TypeError):
            model.setTargets(not_an_array)

    def test__Sets_y(self, model, numpy_array):
        model.setTargets(numpy_array)
        npt.assert_array_equal(model.y, numpy_array)


class Test_evaluateDerivativeOfJ:

    def test__Exists(self, model):
        assert hasattr(model, 'evaluateDerivativeOfJ')

    def test__Throws_If_X_Is_None(self, model):
        with pytest.raises(ValueError):
            model.evaluateDerivativeOfJ()

    def test__Throws_If_y_Is_None(self, model, feature_array):
        model.setFeatures(feature_array)
        with pytest.raises(ValueError):
            model.evaluateDerivativeOfJ()

    def test__Throws_if_m_is_None(self, model, feature_array, target_array):
        model.setFeatures(feature_array)
        model.setTargets(target_array)
        model.m = None
        with pytest.raises(ValueError):
            model.evaluateDerivativeOfJ()

    def test__Throws_if_m_is_zero(self, model, feature_array, target_array):
        model.setFeatures(feature_array)
        model.setTargets(target_array)
        model.m = 0
        with pytest.raises(ValueError):
            model.evaluateDerivativeOfJ()


    def test__Sets_dJ(self, model, feature_array, target_array):
        correct_dJ = [-2.85, -21.45]
        model.setFeatures(feature_array)
        model.setTargets(target_array)
        model.evaluateDerivativeOfJ()
        npt.assert_array_equal(model.dJ, correct_dJ)

class Test__evaluateNewThetas:

    def test__Exists(self, model):
        assert hasattr(model, 'evaluateNewThetas')

    def test__Throws_If_theta_Is_None(self, model, feature_array, target_array):
        model.setFeatures(feature_array)
        model.setTargets(target_array)
        model.theta = None
        with pytest.raises(ValueError):
            model.evaluateNewThetas()

    def test__Throws_If_a_Is_None(self, model, feature_array, target_array):
        model.setFeatures(feature_array)
        model.setTargets(target_array)
        model.a = None
        with pytest.raises(ValueError):
            model.evaluateNewThetas()

    def test__Changes_value_of_theta_0(self, model, feature_array, target_array):
        model.setFeatures(feature_array)
        model.setTargets(target_array)
        initial_theta_0 = model.theta[0]
        model.evaluateNewThetas()
        print("initial theta 0 = ", initial_theta_0)
        print("final theta 0 = ", model.theta[0])
        assert initial_theta_0 != model.theta[0]

    def test__Changes_all_theta_values(self, model, feature_array, target_array):
        model.setFeatures(feature_array)
        model.setTargets(target_array)
        d = np.zeros(len(model.theta))
        initial_theta = model.theta
        model.evaluateNewThetas()
        d = initial_theta - model.theta
        print("initial theta = ", initial_theta)
        print("final theta = ", model.theta)
        print("d = ", d)
        assert d.sum() != 0

class Test__getH:

    def test__Exists(self, model):
        assert hasattr(model, 'getH')

    def test__Returns_numpy_array(self, model, feature_array, target_array):
        model.setFeatures(feature_array)
        model.setTargets(target_array)
        H = model.getH()
        print("theta: ", model.theta)
        print("X:     ", model.X:
        assert isinstance(H, np.ndarray)

