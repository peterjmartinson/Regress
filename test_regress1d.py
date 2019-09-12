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
def Inputs():
    from Regress1D import TrainingInputs
    return TrainingInputs(np.array([[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]]))

@pytest.fixture
def Coefficients():
    from Regress1D import Coefficients
    return Coefficients(2)

@pytest.fixture
def training_inputs():
    return np.array([[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]])

@pytest.fixture
def added_training_example():
    return np.array([3])

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

class Test_Model:

    def test__Set_X_if_X_is_Empty(self, model, training_inputs):
        correct_X = training_inputs
        model.setFeatures(training_inputs)
        print(f'correct_X:  {correct_X}')
        print(f'training_inputs:  {training_inputs}')
        print(f'model.X.training_inputs:  {model.X.training_inputs}')
        npt.assert_array_equal(model.X.training_inputs, correct_X)

    def test__Append_Array_if_X_is_Set(self, model, Inputs, training_inputs, sample_X, added_training_example):
        model.X = Inputs
        model.m = model.X.getNumberOfTrainingExamples()
        model.n = model.X.getNumberOfFeatures()
        correct_X = np.vstack((training_inputs, added_training_example))
        model.setFeatures(added_training_example)
        npt.assert_array_equal(model.X.training_inputs, correct_X)

    def test__Increment_m_if_X_is_Set(self, model, training_inputs, added_training_example):
        model.setFeatures(training_inputs)
        model.setFeatures(added_training_example)
        assert model.m == 11

class Test_setFeatures:

    def test__Exists(self, model):
        assert hasattr(model, 'setFeatures')

    def test__TakesNumpyArray(self, model):
        not_an_array = "not a Numpy ndarray"
        with pytest.raises(TypeError):
            model.setFeatures(not_an_array)

    def test__Set_m_if_X_is_Empty(self, model, training_inputs, m):
        model.setFeatures(training_inputs)
        assert m == model.m

    def test__Throw_Exception_if_Array_is_Wrong_Size(self, model, training_inputs):
        model.X = np.array([1.,1.,1.,1.])
        model.m = 4
        with pytest.raises(TypeError):
            model.setFeatures(training_inputs)

    def test__Set_n_equal_to_1_if_X_is_Empty(self, model, training_inputs):
        model.setFeatures(training_inputs)
        assert model.n == 1

    def test__Set_theta_if_theta_is_not_yet_set(self, model, training_inputs):
        correct_theta = np.array([1, 1])
        model.setFeatures(training_inputs)
        npt.assert_array_equal(model.theta, correct_theta)

class Test_setTargets:

    def test__Exists(self, model):
        assert hasattr(model, 'setTargets')

    def test__Takes_Numpy_Array(self, model):
        not_an_array = "not a Numpy ndarray"
        with pytest.raises(TypeError):
            model.setTargets(not_an_array)

    def test__Throws_if_y_has_not_m_elements(self, model, training_inputs, target_array):
        bad_target_array = np.array([1, 2, 3])
        model.setFeatures(training_inputs)
        with pytest.raises(ValueError):
            model.setTargets(bad_target_array)

    def test__Sets_y(self, model, training_inputs, target_array):
        model.setFeatures(training_inputs)
        model.setTargets(target_array)
        npt.assert_array_equal(model.y, target_array)

class Test_getDerivativeOfJ:

    def test__Exists(self, model):
        assert hasattr(model, 'getDerivativeOfJ')

    def test__Throws_If_X_Is_None(self, model):
        with pytest.raises(ValueError):
            model.getDerivativeOfJ()

    def test__Throws_If_y_Is_None(self, model, training_inputs):
        model.setFeatures(training_inputs)
        with pytest.raises(ValueError):
            model.getDerivativeOfJ()

    def test__Throws_if_m_is_None(self, model, training_inputs, target_array):
        model.setFeatures(training_inputs)
        model.setTargets(target_array)
        model.m = None
        with pytest.raises(ValueError):
            model.getDerivativeOfJ()

    def test__Throws_if_m_is_zero(self, model, training_inputs, target_array):
        model.setFeatures(training_inputs)
        model.setTargets(target_array)
        model.m = 0
        with pytest.raises(ValueError):
            model.getDerivativeOfJ()

    def test_Returns_an_appropriate_derivative(self, model, training_inputs, target_array):
        correct_dJ = [-2.85, -21.45]
        model.setFeatures(training_inputs)
        model.setTargets(target_array)
        print(f'X:  {model.X.getTrainingInputs()}')
        print(f'theta:  {model.theta}')
        print(f'self.m:  {model.m}')
        print(f'self.getHypothesis():  {model.getHypothesis()}')
        print(f'self.y:  {model.y}')
        print(f'derivative of J:  {model.getDerivativeOfJ()}')
        result_dJ = model.getDerivativeOfJ()
        npt.assert_array_equal(result_dJ, correct_dJ)

    def test__Changes_value_of_theta_0(self, model, training_inputs, target_array):
        model.setFeatures(training_inputs)
        model.setTargets(target_array)
        initial_theta_0 = model.theta[0]
        print("initial theta 0 = ", initial_theta_0)
        model.evaluateNewThetas()
        print("final theta 0 = ", model.theta[0])
        assert initial_theta_0 != model.theta[0]

    def test__Changes_all_theta_values(self, model, training_inputs, target_array):
        model.setFeatures(training_inputs)
        model.setTargets(target_array)
        d = np.zeros(len(model.theta))
        initial_theta = model.theta
        print("initial theta = ", initial_theta)
        model.evaluateNewThetas()
        d = initial_theta - model.theta
        print("final theta = ", model.theta)
        print("d = ", d)
        assert d.sum() != 0


class Test__evaluateNewThetas:

    def test__Exists(self, model):
        assert hasattr(model, 'evaluateNewThetas')

    def test__Throws_If_theta_Is_None(self, model, training_inputs, target_array):
        model.setFeatures(training_inputs)
        model.setTargets(target_array)
        model.theta = None
        with pytest.raises(ValueError):
            model.evaluateNewThetas()

    def test__Throws_If_a_Is_None(self, model, training_inputs, target_array):
        model.setFeatures(training_inputs)
        model.setTargets(target_array)
        model.a = None
        with pytest.raises(ValueError):
            model.evaluateNewThetas()

class Test__getJ:

    def test__Exists(self, model):
        assert hasattr(model, 'getJ')

    def test__Returns_a_float(self, model, training_inputs, target_array):
        model.setFeatures(training_inputs)
        model.setTargets(target_array)
        J = model.getJ()
        assert isinstance(J, float)

    def test__Returns_correct_J(self, model, training_inputs, target_array):
        model.setFeatures(training_inputs)
        model.setTargets(target_array)
        correct_J = 6.082499999999999
        J = model.getJ()
        assert J == correct_J






class Test__getHypothesis:

    def test__Exists(self, model):
        assert hasattr(model, 'getHypothesis')

    def test__Returns_numpy_array(self, model, training_inputs, target_array):
        model.setFeatures(training_inputs)
        model.setTargets(target_array)
        H = model.getHypothesis()
        assert isinstance(H, np.ndarray)

    def test__Resulting_hypothesis_is_1_by_m(self, model, training_inputs, target_array):
        model.setFeatures(training_inputs)
        model.setTargets(target_array)
        h = model.getHypothesis()
        m = model.m
        assert h.size == m


class Test__Class_ResidualSumOfSquares:

    def test__Exists(self, RSS):
        assert RSS != None

    def test__getValue_exists(self, RSS):
        assert hasattr(RSS, 'getValue')

    def test__getValue_returns_a_float(self, RSS, model, training_inputs, target_array, hypothesis):
        J = RSS.getValue(hypothesis, target_array)
        assert isinstance(J, float)

    def test__getValue_returns_correct_J(self, RSS, model, training_inputs, target_array, hypothesis):
        correct_J = 6.082499999999999
        J = RSS.getValue(hypothesis, target_array)
        assert J == correct_J


class Test__Class_TrainingInputs:

    def test__Initializing_takes_numpy_array(self):
        from Regress1D import TrainingInputs
        not_an_array = "not a Numpy ndarray"
        with pytest.raises(TypeError):
            Inputs = TrainingInputs(not_an_array)

    def test__Initializing_sets_training_inputs(self, training_inputs):
        correct_training_set = training_inputs
        from Regress1D import TrainingInputs
        Inputs = TrainingInputs(training_inputs)
        npt.assert_array_equal(Inputs.training_inputs, correct_training_set)

    def test__Initializing_sets_number_of_features(self, training_inputs):
        correct_n = 1
        from Regress1D import TrainingInputs
        Inputs = TrainingInputs(training_inputs)
        assert Inputs.number_of_features == correct_n

    def test__Initializing_sets_number_of_training_examples(self, training_inputs):
        correct_m = 10
        from Regress1D import TrainingInputs
        Inputs = TrainingInputs(training_inputs)
        assert Inputs.number_of_training_examples == correct_m

    def test__getTrainingInputs_gets_the_right_inputs(self, Inputs, training_inputs):
        gotten_inputs = Inputs.getTrainingInputs()
        npt.assert_array_equal(gotten_inputs, training_inputs)

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

    def test__addTrainingExample_adds_correct_training_example(self, Inputs, training_inputs):
        initial_array = training_inputs
        added_training_example = np.array([3])
        correct_array = np.vstack((initial_array, added_training_example))
        Inputs.addTrainingExample(added_training_example)
        npt.assert_array_equal(correct_array, Inputs.training_inputs)

    def test__addTrainingExample_increments_number_of_training_examples(self, Inputs):
        added_training_example = np.array([3])
        correct_number_of_training_examples = 11
        Inputs.addTrainingExample(added_training_example)
        assert Inputs.number_of_training_examples == correct_number_of_training_examples
        

## Your feature_array is the wrong shape
## instead of (1,10), it should be (10,1)!!
## if there is only one feature, array should be [[1],[2],[3],...]
## if there are two or more features, should be [[1,1],[2,2],[3,3],...]


class Test_Class_Coefficients:

    def test__Initializing_takes_an_integer(self):
        from Regress1D import Coefficients
        not_an_integer = "not an integer"
        with pytest.raises(TypeError):
            Coef = Coefficients(not_an_integer)
    
    def test__Initializing_sets_appropriate_array(self):
        from Regress1D import Coefficients
        n = 2
        correct_array = np.array([1., 1.])
        Coefficients = Coefficients(n)
        npt.assert_array_equal(Coefficients.c, correct_array)

    def test__Has_method_getCoefficients(self, Coefficients):
        assert hasattr(Coefficients, 'getCoefficients')

    def test__getCoefficients_returns_correct_array(self):
        from Regress1D import Coefficients
        n = 2
        correct_array = np.array([1., 1.])
        Coefficients = Coefficients(n)
        c = Coefficients.getCoefficients()
        npt.assert_array_equal(c, correct_array)

    def test__Has_method_addCoefficient(self, Coefficients):
        assert hasattr(Coefficients, 'addCoefficient')

    def test__addCoefficient_returns_coefficients_array(self, Coefficients):
        output = Coefficients.addCoefficient()
        correct_array = Coefficients.c
        npt.assert_array_equal(output, correct_array)

    def test_addCoefficient_adds_one_element(self, Coefficients):
        Coefficients.addCoefficient()
        c = Coefficients.c
        correct_array = np.array([1., 1., 1.])
        npt.assert_array_equal(c, correct_array)

    def test__Has_method_updateCoefficient(self, Coefficients):
        assert hasattr(Coefficients, 'updateCoefficient')

    def test__updateCoefficient_throws_if_index_out_of_range(self, Coefficients):
        with pytest.raises(ValueError):
            Coefficients.updateCoefficient(10, 1.)

    def test__updateCoefficient_throws_if_replacement_element_not_a_float(self, Coefficients):
        with pytest.raises(TypeError):
            Coefficients.updateCoefficient(1, 1)

    def test__updateCoefficient_returns_new_array(self, Coefficients):
        replacement_element = 2.
        correct_output = np.array([1., 2.])
        output = Coefficients.updateCoefficient(1, replacement_element)
        npt.assert_array_equal(output, correct_output)












