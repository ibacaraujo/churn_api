"""
Unit tests for the ML model - no server required!
"""
import pickle
import pytest


@pytest.fixture
def model():
    """Load the trained model."""
    with open('model.bin', 'rb') as f:
        return pickle.load(f)


@pytest.fixture
def sample_customer():
    """Sample customer data for testing."""
    return {
        'gender': 'female',
        'seniorcitizen': 0,
        'partner': 'yes',
        'dependents': 'no',
        'phoneservice': 'no',
        'multiplelines': 'no_phone_service',
        'internetservice': 'dsl',
        'onlinesecurity': 'no',
        'onlinebackup': 'yes',
        'deviceprotection': 'no',
        'techsupport': 'no',
        'streamingtv': 'no',
        'streamingmovies': 'no',
        'contract': 'month-to-month',
        'paperlessbilling': 'yes',
        'paymentmethod': 'electronic_check',
        'tenure': 1,
        'monthlycharges': 29.85,
        'totalcharges': 29.85,
    }


@pytest.fixture
def loyal_customer():
    """Long-term loyal customer - should have low churn probability."""
    return {
        'gender': 'male',
        'seniorcitizen': 0,
        'partner': 'yes',
        'dependents': 'yes',
        'phoneservice': 'yes',
        'multiplelines': 'yes',
        'internetservice': 'fiber_optic',
        'onlinesecurity': 'yes',
        'onlinebackup': 'yes',
        'deviceprotection': 'yes',
        'techsupport': 'yes',
        'streamingtv': 'yes',
        'streamingmovies': 'yes',
        'contract': 'two_year',
        'paperlessbilling': 'no',
        'paymentmethod': 'bank_transfer_(automatic)',
        'tenure': 72,  # 6 years!
        'monthlycharges': 110.0,
        'totalcharges': 7920.0,
    }


class TestModelLoading:
    """Tests for model loading and structure."""
    
    def test_model_loads_successfully(self, model):
        """Model file exists and loads without errors."""
        assert model is not None
    
    def test_model_has_predict_proba(self, model):
        """Model has the predict_proba method we need."""
        assert hasattr(model, 'predict_proba')
    
    def test_model_has_predict(self, model):
        """Model has the predict method."""
        assert hasattr(model, 'predict')


class TestPredictions:
    """Tests for model predictions."""
    
    def test_prediction_returns_probability(self, model, sample_customer):
        """Prediction returns a value between 0 and 1."""
        prob = model.predict_proba([sample_customer])[0, 1]
        assert 0.0 <= prob <= 1.0
    
    def test_prediction_returns_binary(self, model, sample_customer):
        """Predict returns 0 or 1."""
        pred = model.predict([sample_customer])[0]
        assert pred in [0, 1]
    
    def test_high_risk_customer_prediction(self, model, sample_customer):
        """New customer with month-to-month contract should have higher churn risk."""
        prob = model.predict_proba([sample_customer])[0, 1]
        # Month-to-month with low tenure = higher risk
        assert prob > 0.3, f"Expected higher churn risk, got {prob}"
    
    def test_loyal_customer_prediction(self, model, loyal_customer):
        """Long-term customer with 2-year contract should have lower churn risk."""
        prob = model.predict_proba([loyal_customer])[0, 1]
        # Two-year contract with high tenure = lower risk
        assert prob < 0.5, f"Expected lower churn risk for loyal customer, got {prob}"


class TestModelConsistency:
    """Tests for model behavior consistency."""
    
    def test_same_input_same_output(self, model, sample_customer):
        """Same input should always produce the same output."""
        prob1 = model.predict_proba([sample_customer])[0, 1]
        prob2 = model.predict_proba([sample_customer])[0, 1]
        assert prob1 == prob2
    
    def test_batch_prediction(self, model, sample_customer, loyal_customer):
        """Model can handle batch predictions."""
        probs = model.predict_proba([sample_customer, loyal_customer])
        assert probs.shape == (2, 2)  # 2 samples, 2 classes
