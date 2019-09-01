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
    from Regress1D import Model
    return Model()

@pytest.fixture
def RSS():
    from Regress1D import ResidualSumOfSquares
    return ResidualSumOfSquares()

@pytest.fixture
def numpy_array():
    return np.array([1., 2., 3.])

@pytest.fixture
def feature_array():
    return np.array([1,2,3,4,5,6,7,8,9,10])

@pytest.fixture
def Inputs():
    from Regress1D import TrainingInputs
    return TrainingInputs(np.array([1,2,3,4,5,6,7,8,9,10]))

@pytest.fixture
def sample_X():
    return np.array([[1,1,1,1,1,1,1,1,1,1], [1,2,3,4,5,6,7,8,9,10]])

@pytest.fixture
def target_array():
    feature_array = np.array([1,2,3,4,5,6,7,8,9,10])
    return feature_array * 1.7

@pytest.fixture
def m():
    return 10

@pytest.fixture
def hypothesis():
    return np.array([2., 3., 4., 5., 6., 7., 8., 9., 10., 11.])

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

    def test__Set_X_if_X_is_Empty(self, model, feature_array, sample_X):
        correct_X = sample_X
        model.setFeatures(feature_array)
        npt.assert_array_equal(model.X, correct_X)

    def test__Set_m_if_X_is_Empty(self, model, feature_array, m):
        model.setFeatures(feature_array)
        assert m == model.m

    def test__Throw_Exception_if_Array_is_Wrong_Size(self, model, feature_array):
        model.X = np.array([1.,1.,1.,1.])
        model.m = 4
        with pytest.raises(TypeError):
            model.setFeatures(feature_array)

    def test__Append_Array_if_X_is_Set(self, model, feature_array, sample_X):
        model.X = sample_X
        model.m = 10
        correct_X = np.vstack((sample_X, feature_array))
        model.setFeatures(feature_array)
        npt.assert_array_equal(model.X, correct_X)

    def test__Set_n_equal_to_1_if_X_is_Empty(self, model, feature_array):
        model.setFeatures(feature_array)
        assert model.n == 1

    def test__Increment_n_if_X_is_Set(self, model, feature_array):
        model.setFeatures(feature_array)
        model.setFeatures(feature_array)
        assert model.n == 2

    def test__Set_theta_if_theta_is_not_yet_set(self, model, feature_array):
        correct_theta = np.array([1, 1])
        model.setFeatures(feature_array)
        npt.assert_array_equal(model.theta, correct_theta)

    def test__Add_another_theta_if_theta_is_already_set(self, model, feature_array):
        correct_theta = np.array([1,1,1,1])
        model.setFeatures(feature_array)
        model.setFeatures(feature_array)
        model.setFeatures(feature_array)
        npt.assert_array_equal(model.theta, correct_theta)


class Test_setTargets:

    def test__Exists(self, model):
        assert hasattr(model, 'setTargets')

    def test__Takes_Numpy_Array(self, model):
        not_an_array = "not a Numpy ndarray"
        with pytest.raises(TypeError):
            model.setTargets(not_an_array)

    def test__Throws_if_y_has_not_m_elements(self, model, feature_array, target_array):
        bad_target_array = np.array([1, 2, 3])
        model.setFeatures(feature_array)
        with pytest.raises(ValueError):
            model.setTargets(bad_target_array)

    def test__Sets_y(self, model, feature_array, target_array):
        model.setFeatures(feature_array)
        model.setTargets(target_array)
        npt.assert_array_equal(model.y, target_array)

class Test_getDerivativeOfJ:

    def test__Exists(self, model):
        assert hasattr(model, 'getDerivativeOfJ')

    def test__Throws_If_X_Is_None(self, model):
        with pytest.raises(ValueError):
            model.getDerivativeOfJ()

    def test__Throws_If_y_Is_None(self, model, feature_array):
        model.setFeatures(feature_array)
        with pytest.raises(ValueError):
            model.getDerivativeOfJ()

    def test__Throws_if_m_is_None(self, model, feature_array, target_array):
        model.setFeatures(feature_array)
        model.setTargets(target_array)
        model.m = None
        with pytest.raises(ValueError):
            model.getDerivativeOfJ()

    def test__Throws_if_m_is_zero(self, model, feature_array, target_array):
        model.setFeatures(feature_array)
        model.setTargets(target_array)
        model.m = 0
        with pytest.raises(ValueError):
            model.getDerivativeOfJ()

    def test_Returns_an_appropriate_derivative(self, model, feature_array, target_array):
        correct_dJ = [-2.85, -21.45]
        model.setFeatures(feature_array)
        model.setTargets(target_array)
        result_dJ = model.getDerivativeOfJ()
        npt.assert_array_equal(result_dJ, correct_dJ)


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

class Test__getJ:

    def test__Exists(self, model):
        assert hasattr(model, 'getJ')

    def test__Returns_a_float(self, model, feature_array, target_array):
        model.setFeatures(feature_array)
        model.setTargets(target_array)
        J = model.getJ()
        assert isinstance(J, float)

    def test__Returns_correct_J(self, model, feature_array, target_array):
        model.setFeatures(feature_array)
        model.setTargets(target_array)
        correct_J = 6.082499999999999
        J = model.getJ()
        assert J == correct_J






class Test__getHypothesis:

    def test__Exists(self, model):
        assert hasattr(model, 'getHypothesis')

    def test__Returns_numpy_array(self, model, feature_array, target_array):
        model.setFeatures(feature_array)
        model.setTargets(target_array)
        H = model.getHypothesis()
        assert isinstance(H, np.ndarray)

    def test__Resulting_hypothesis_is_1_by_m(self, model, feature_array, target_array):
        model.setFeatures(feature_array)
        model.setTargets(target_array)
        h = model.getHypothesis()
        m = model.m
        assert h.size == m


class Test__Class_ResidualSumOfSquares:

    def test__Exists(self, RSS):
        assert RSS != None

    def test__getValue_exists(self, RSS):
        assert hasattr(RSS, 'getValue')

    def test__getValue_returns_a_float(self, RSS, model, feature_array, target_array, hypothesis):
        J = RSS.getValue(hypothesis, target_array)
        assert isinstance(J, float)

    def test__getValue_returns_correct_J(self, RSS, model, feature_array, target_array, hypothesis):
        correct_J = 6.082499999999999
        J = RSS.getValue(hypothesis, target_array)
        assert J == correct_J


class Test__Class_TrainingInputs:

    # def test__Exists(self, Inputs):
    #     assert Inputs != None

    # def test__setInputs_exists(self, Inputs):
    #     assert hasattr(Inputs, 'setInputs')

    # def test__Exists(self, model):
    #     assert hasattr(model, 'setFeatures')

    def test__Initializing_takes_numpy_array(self):
        from Regress1D import TrainingInputs
        not_an_array = "not a Numpy ndarray"
        with pytest.raises(TypeError):
            Inputs = TrainingInputs(not_an_array)

    def test__Initializing_sets_training_inputs(self, feature_array):
        x_0 = np.ones(10)
        correct_training_set = feature_array
        from Regress1D import TrainingInputs
        Inputs = TrainingInputs(feature_array)
        npt.assert_array_equal(Inputs.training_inputs, correct_training_set)

    def test__Initializing_sets_number_of_features(self, feature_array):
        correct_n = 1
        from Regress1D import TrainingInputs
        Inputs = TrainingInputs(feature_array)
        assert Inputs.number_of_features == correct_n

    def test__Initializing_sets_number_of_training_examples(self, feature_array):
        correct_m = 10
        from Regress1D import TrainingInputs
        Inputs = TrainingInputs(feature_array)
        assert Inputs.number_of_training_examples == correct_m


    def test__getTrainingInputs_gets_the_right_inputs(self, Inputs, feature_array):
        gotten_inputs = Inputs.getTrainingInputs()
        npt.assert_array_equal(gotten_inputs, feature_array)

    def test__getNumberOfTrainingExamples_gets_the_right_number(self, Inputs, m):
        gotten_number = Inputs.getNumberOfTrainingExamples()
        assert gotten_number == m

    def test__getNumberOfFeatures_gets_the_right_number(self, Inputs):
        correct_n = 1
        gotten_number = Inputs.getNumberOfFeatures()
        assert gotten_number == correct_n

    def test__addTrainingExample_takes_a_numpy_array(self, Inputs):
        not_an_array = "not a Numpy arry"
        with pytest.raises(TypeError):
            new_inputs = Inputs.addTrainingExample(not_an_array)

    def test__addTrainingExample_throws_if_array_is_wrong_size(self, Inputs):
        wrong_array = np.array([1,2,3])
        with pytest.raises(ValueError):
            new_inputs = Inputs.addTrainingExample(wrong_array)

## Your feature_array is the wrong shape
## instead of (1,10), it should be (10,1)!!












    # def test__Set_m_if_X_is_Empty(self, model, feature_array, m):
    #     model.setFeatures(feature_array)
    #     assert m == model.m

    # def test__Throw_Exception_if_Array_is_Wrong_Size(self, model, feature_array):
    #     model.X = np.array([1.,1.,1.,1.])
    #     model.m = 4
    #     with pytest.raises(TypeError):
    #         model.setFeatures(feature_array)

    # def test__Append_Array_if_X_is_Set(self, model, feature_array, sample_X):
    #     model.X = sample_X
    #     model.m = 10
    #     correct_X = np.vstack((sample_X, feature_array))
    #     model.setFeatures(feature_array)
    #     npt.assert_array_equal(model.X, correct_X)

    # def test__Set_n_equal_to_1_if_X_is_Empty(self, model, feature_array):
    #     model.setFeatures(feature_array)
    #     assert model.n == 1

    # def test__Increment_n_if_X_is_Set(self, model, feature_array):
    #     model.setFeatures(feature_array)
    #     model.setFeatures(feature_array)
    #     assert model.n == 2

    # def test__Set_theta_if_theta_is_not_yet_set(self, model, feature_array):
    #     correct_theta = np.array([1, 1])
    #     model.setFeatures(feature_array)
    #     npt.assert_array_equal(model.theta, correct_theta)

    # def test__Add_another_theta_if_theta_is_already_set(self, model, feature_array):
    #     correct_theta = np.array([1,1,1,1])
    #     model.setFeatures(feature_array)
    #     model.setFeatures(feature_array)
    #     model.setFeatures(feature_array)
    #     npt.assert_array_equal(model.theta, correct_theta)
