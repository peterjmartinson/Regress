import pytest
import numpy as np
import numpy.testing as npt

def printDiagnostics(self):
    print("--- Diagnostic ---")
    print("X (Feature array):               ", self.X)
    print("y (Target array):                ", self.y)
    print("m (Number of training examples): ", self.m)
    print("n (Number of features):          ", self.n)
    print("beta (Parameter array):         ", self.beta)
    print("dJ (container for all current dJ/dBeta values): ", self.dJ)
    print("a (Learning rate):               ", self.a)

## ======================================================= Pytest Fixtures

@pytest.fixture
def Predictors():
    from Regress1D import TrainingPredictors
    return TrainingPredictors(np.array([[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]]))

@pytest.fixture
def Responses():
    from Regress1D import TrainingResponses
    return TrainingResponses(np.array([1,2,3,4,5,6,7,8,9,10]) * 1.7)

@pytest.fixture
def Coefficients():
    from Regress1D import Coefficients
    return Coefficients(2)

@pytest.fixture
def training_predictors():
    return np.array([[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]])

@pytest.fixture
def added_training_predictor():
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
def predictor_array():
    return np.array([1,2,3,4,5,6,7,8,9,10])

@pytest.fixture
def sample_X():
    return np.array([[1,1,1,1,1,1,1,1,1,1], [1,2,3,4,5,6,7,8,9,10]])

@pytest.fixture
def training_responses():
    predictor_array = np.array([1,2,3,4,5,6,7,8,9,10])
    return predictor_array * 1.7

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

    def test__Set_X_if_X_is_Empty(self, model, training_predictors):
        correct_X = training_predictors
        model.setPredictors(training_predictors)
        print(f'correct_X:  {correct_X}')
        print(f'training_predictors:  {training_predictors}')
        print(f'model.X.training_predictors:  {model.X.training_predictors}')
        npt.assert_array_equal(model.X.training_predictors, correct_X)

    def test__Append_Array_if_X_is_Set(self, model, Predictors, training_predictors, sample_X, added_training_predictor):
        model.X = Predictors
        model.m = model.X.getNumberOfTrainingPredictors()
        model.n = model.X.getNumberOfFeatures()
        correct_X = np.vstack((training_predictors, added_training_predictor))
        model.setPredictors(added_training_predictor)
        npt.assert_array_equal(model.X.training_predictors, correct_X)

    def test__Increment_m_if_X_is_Set(self, model, training_predictors, added_training_predictor):
        model.setPredictors(training_predictors)
        model.setPredictors(added_training_predictor)
        assert model.m == 11

class Test_setPredictors:

    def test__Exists(self, model):
        assert hasattr(model, 'setPredictors')

    def test__TakesNumpyArray(self, model):
        not_an_array = "not a Numpy ndarray"
        with pytest.raises(TypeError):
            model.setPredictors(not_an_array)

    def test__Set_m_if_X_is_Empty(self, model, training_predictors, m):
        model.setPredictors(training_predictors)
        assert m == model.m

    def test__Throw_Exception_if_Array_is_Wrong_Size(self, model, training_predictors):
        model.X = np.array([1.,1.,1.,1.])
        model.m = 4
        with pytest.raises(TypeError):
            model.setPredictors(training_predictors)

    def test__Set_n_equal_to_1_if_X_is_Empty(self, model, training_predictors):
        model.setPredictors(training_predictors)
        assert model.n == 1

    def test__Set_beta_if_beta_is_not_yet_set(self, model, training_predictors):
        correct_beta = np.array([1, 1])
        model.setPredictors(training_predictors)
        npt.assert_array_equal(model.beta.getCoefficients(), correct_beta)

class Test_setTargets:

    def test__Exists(self, model):
        assert hasattr(model, 'setTargets')

    def test__Takes_Numpy_Array(self, model):
        not_an_array = "not a Numpy ndarray"
        with pytest.raises(TypeError):
            model.setTargets(not_an_array)

    def test__Throws_if_y_has_not_m_elements(self, model, training_predictors, training_responses):
        bad_training_responses = np.array([1, 2, 3])
        model.setPredictors(training_predictors)
        with pytest.raises(ValueError):
            model.setTargets(bad_training_responses)

    def test__Sets_y(self, model, training_predictors, training_responses):
        model.setPredictors(training_predictors)
        model.setTargets(training_responses)
        npt.assert_array_equal(model.y.getTrainingResponses(), training_responses)

class Test__evaluateNewBetas:

    def test__Exists(self, model):
        assert hasattr(model, 'evaluateNewBetas')

    def test__Throws_If_beta_Is_None(self, model, training_predictors, training_responses):
        model.setPredictors(training_predictors)
        model.setTargets(training_responses)
        model.beta = None
        with pytest.raises(ValueError):
            model.evaluateNewBetas()

    def test__Throws_If_a_Is_None(self, model, training_predictors, training_responses):
        model.setPredictors(training_predictors)
        model.setTargets(training_responses)
        model.a = None
        with pytest.raises(ValueError):
            model.evaluateNewBetas()

    def test__Changes_value_of_beta_0(self, model, training_predictors, training_responses):
        model.setPredictors(training_predictors)
        model.setTargets(training_responses)
        initial_beta_0 = model.beta.getCoefficients()[0]
        print("initial beta 0 = ", initial_beta_0)
        model.evaluateNewBetas()
        print("final beta 0 = ", model.beta.getCoefficients()[0])
        assert initial_beta_0 != model.beta.getCoefficients()[0]

    def test__Changes_all_beta_values(self, model, training_predictors, training_responses):
        model.setPredictors(training_predictors)
        model.setTargets(training_responses)
        d = np.ones(len(model.beta.getCoefficients()))
        model.evaluateNewBetas()
        d -= model.beta.getCoefficients()
        print("final beta = ", model.beta.getCoefficients())
        print("d = ", d)
        assert d.sum() != 0.0

class Test__getHypothesis:

    def test__Exists(self, model):
        assert hasattr(model, 'getHypothesis')

    def test__Returns_numpy_array(self, model, training_predictors, training_responses):
        model.setPredictors(training_predictors)
        model.setTargets(training_responses)
        H = model.getHypothesis()
        assert isinstance(H, np.ndarray)

    def test__Resulting_hypothesis_is_1_by_m(self, model, training_predictors, training_responses):
        model.setPredictors(training_predictors)
        model.setTargets(training_responses)
        h = model.getHypothesis()
        m = model.m
        assert h.size == m


class Test__Class_ResidualSumOfSquares:

    def test__Exists(self, RSS):
        assert RSS != None

    def test__getValue_exists(self, RSS):
        assert hasattr(RSS, 'getValue')

    def test__getValue_returns_a_float(self, RSS, model, training_predictors, training_responses, hypothesis):
        J = RSS.getValue(training_responses, hypothesis)
        assert isinstance(J, float)

    def test__getValue_returns_correct_J(self, RSS, model, training_predictors, training_responses, hypothesis):
        correct_J = 6.082499999999999
        J = RSS.getValue(training_responses, hypothesis)
        assert J == correct_J

    def test_getDerivative_returns_an_appropriate_derivative(self, RSS, training_predictors, training_responses, hypothesis):
        correct_dJ = [-2.85, -21.45]
        print(f'X:  {training_predictors}')
        print(f'self.m:  {len(training_responses)}')
        print(f'hypothesis:  {hypothesis}')
        print(f'y:  {training_responses}')
        result_dJ = RSS.getDerivative(training_predictors, training_responses, hypothesis)
        print(f'derivative of J:  {result_dJ}')
        print(f'expected dJ: {correct_dJ}')
        npt.assert_array_equal(result_dJ, correct_dJ)

    def test__Throws_if_predictor_is_not_numpy_array(self, RSS, training_responses, hypothesis):
        training_predictors = 'not an array'
        with pytest.raises(TypeError):
            RSS.getDerivative(training_predictors, training_responses, hypothesis)

    def test__Throws_if_target_is_not_numpy_array(self, RSS, training_predictors, hypothesis):
        training_responses = 'not an array'
        with pytest.raises(TypeError):
            RSS.getDerivative(training_predictors, training_responses, hypothesis)

    def test__Throws_if_target_is_length_zero(self, RSS, training_predictors, hypothesis):
        training_responses = np.array([])
        print(f'm = {len(training_responses)}')
        with pytest.raises(ZeroDivisionError):
            RSS.getDerivative(training_predictors, training_responses, hypothesis)

    def test__Throws_if_hypothesis_is_not_numpy_array(self, RSS, training_predictors, training_responses):
        hypothesis = 'not an array'
        with pytest.raises(TypeError):
            RSS.getDerivative(training_predictors, training_responses, hypothesis)

class Test__Class_TrainingPredictors:

    def test__Initializing_takes_numpy_array(self):
        from Regress1D import TrainingPredictors
        not_an_array = "not a Numpy ndarray"
        with pytest.raises(TypeError):
            Predictors = TrainingPredictors(not_an_array)

    def test__Initializing_sets_training_predictors(self, training_predictors):
        correct_training_set = training_predictors
        from Regress1D import TrainingPredictors
        Predictors = TrainingPredictors(training_predictors)
        npt.assert_array_equal(Predictors.training_predictors, correct_training_set)

    def test__Initializing_sets_number_of_features(self, training_predictors):
        correct_n = 1
        from Regress1D import TrainingPredictors
        Predictors = TrainingPredictors(training_predictors)
        assert Predictors.number_of_features == correct_n

    def test__Initializing_sets_number_of_training_predictors(self, training_predictors):
        correct_m = 10
        from Regress1D import TrainingPredictors
        Predictors = TrainingPredictors(training_predictors)
        assert Predictors.number_of_training_predictors == correct_m

    def test__getTrainingPredictors_gets_the_right_predictors(self, Predictors, training_predictors):
        gotten_predictors = Predictors.getTrainingPredictors()
        npt.assert_array_equal(gotten_predictors, training_predictors)

    def test__getNumberOfTrainingPredictors_gets_the_right_number(self, Predictors, m):
        gotten_number = Predictors.getNumberOfTrainingPredictors()
        assert gotten_number == m

    def test__getNumberOfFeatures_gets_the_right_number(self, Predictors):
        correct_n = 1
        gotten_number = Predictors.getNumberOfFeatures()
        assert gotten_number == correct_n

    def test__addTrainingPredictor_takes_a_numpy_array(self, Predictors):
        not_an_array = "not a Numpy arry"
        with pytest.raises(TypeError):
            new_predictors = Predictors.addTrainingPredictor(not_an_array)

    def test__addTrainingPredictor_throws_if_array_is_wrong_size(self, Predictors):
        wrong_array = np.array([1,2,3])
        with pytest.raises(ValueError):
            new_predictors = Predictors.addTrainingPredictor(wrong_array)

    def test__addTrainingPredictor_adds_correct_training_predictor(self, Predictors, training_predictors):
        initial_array = training_predictors
        added_training_predictor = np.array([3])
        correct_array = np.vstack((initial_array, added_training_predictor))
        Predictors.addTrainingPredictor(added_training_predictor)
        npt.assert_array_equal(correct_array, Predictors.training_predictors)

    def test__addTrainingPredictor_increments_number_of_training_predictors(self, Predictors):
        added_training_predictor = np.array([3])
        correct_number_of_training_predictors = 11
        Predictors.addTrainingPredictor(added_training_predictor)
        assert Predictors.number_of_training_predictors == correct_number_of_training_predictors
        

class Test__Class_TrainingResponses:

    def test__Initializing_takes_numpy_array(self):
        from Regress1D import TrainingResponses
        not_an_array = "not a Numpy ndarray"
        with pytest.raises(TypeError):
            Predictors = TrainingResponses(not_an_array)

    def test__Initializing_sets_training_responses(self, training_responses):
        correct_training_set = training_responses
        from Regress1D import TrainingResponses
        Responses = TrainingResponses(training_responses)
        npt.assert_array_equal(Responses.training_responses, correct_training_set)

    def test__getTrainingResponses_gets_the_right_responses(self, Responses, training_responses):
        gotten_responses = Responses.getTrainingResponses()
        npt.assert_array_equal(gotten_responses, training_responses)

    def test__addTrainingResponse_takes_a_numpy_array(self, Responses):
        not_an_array = "not a Numpy arry"
        with pytest.raises(TypeError):
            new_responses = Responses.addTrainingResponse(not_an_array)

    def test__addTrainingResponse_adds_correct_training_response(self, Responses, training_responses):
        initial_array = training_responses
        added_training_response = np.array([3])
        correct_array = np.append(initial_array, added_training_response)
        Responses.addTrainingResponse(added_training_response)
        npt.assert_array_equal(correct_array, Responses.training_responses)

    def test__addTrainingResponse_adds_correct_training_response(self, Responses, training_responses):
        initial_array = training_responses
        added_training_response = np.array([3])
        correct_array = np.append(initial_array, added_training_response)
        returned_array = Responses.addTrainingResponse(added_training_response)
        npt.assert_array_equal(correct_array, returned_array)



class Test_Class_Coefficients:

    def test__Initializing_takes_an_integer(self):
        from Regress1D import Coefficients
        not_an_integer = "not an integer"
        with pytest.raises(TypeError):
            Coef = Coefficients(not_an_integer)
    
    def test__Initializing_sets_appropriate_array(self):
        from Regress1D import Coefficients
        n = 2
        correct_array = np.array([1., 1., 1.])
        Coefficients = Coefficients(n)
        npt.assert_array_equal(Coefficients.c, correct_array)

    def test__Has_method_getCoefficients(self, Coefficients):
        assert hasattr(Coefficients, 'getCoefficients')

    def test__getCoefficients_returns_correct_array(self):
        from Regress1D import Coefficients
        n = 2
        correct_array = np.array([1., 1., 1.])
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
        correct_array = np.array([1., 1., 1., 1.])
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
        correct_output = np.array([1., 2., 1.])
        output = Coefficients.updateCoefficient(1, replacement_element)
        npt.assert_array_equal(output, correct_output)










