"""
API tests using FastAPI TestClient - no server required!
"""
import pytest
from fastapi.testclient import TestClient

from predict import app


@pytest.fixture
def client():
    """Create a test client for the API."""
    return TestClient(app)


@pytest.fixture
def valid_customer():
    """Valid customer payload."""
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


class TestHealthEndpoints:
    """Test health and monitoring endpoints."""
    
    def test_docs_available(self, client):
        """OpenAPI docs should be available."""
        response = client.get("/docs")
        assert response.status_code == 200
    
    def test_metrics_endpoint(self, client):
        """Prometheus metrics endpoint should work."""
        response = client.get("/metrics")
        assert response.status_code == 200
        assert "python_info" in response.text


class TestPredictionEndpoint:
    """Test the /predict endpoint."""
    
    def test_predict_success(self, client, valid_customer):
        """Valid customer should return prediction."""
        response = client.post("/predict", json=valid_customer)
        assert response.status_code == 200
        
        data = response.json()
        assert "churn_probability" in data
        assert "churn" in data
        assert isinstance(data["churn_probability"], float)
        assert isinstance(data["churn"], bool)
    
    def test_predict_probability_range(self, client, valid_customer):
        """Probability should be between 0 and 1."""
        response = client.post("/predict", json=valid_customer)
        data = response.json()
        assert 0.0 <= data["churn_probability"] <= 1.0
    
    def test_predict_churn_consistency(self, client, valid_customer):
        """Churn boolean should match probability threshold."""
        response = client.post("/predict", json=valid_customer)
        data = response.json()
        
        expected_churn = data["churn_probability"] >= 0.5
        assert data["churn"] == expected_churn


class TestInputValidation:
    """Test input validation (Pydantic + Pandera)."""
    
    def test_missing_field(self, client, valid_customer):
        """Missing required field should return 422."""
        del valid_customer["gender"]
        response = client.post("/predict", json=valid_customer)
        assert response.status_code == 422
    
    def test_invalid_gender(self, client, valid_customer):
        """Invalid gender value should return 422."""
        valid_customer["gender"] = "invalid"
        response = client.post("/predict", json=valid_customer)
        assert response.status_code == 422
    
    def test_invalid_contract(self, client, valid_customer):
        """Invalid contract type should return 422."""
        valid_customer["contract"] = "invalid_contract"
        response = client.post("/predict", json=valid_customer)
        assert response.status_code == 422
    
    def test_negative_tenure(self, client, valid_customer):
        """Negative tenure should return 422."""
        valid_customer["tenure"] = -1
        response = client.post("/predict", json=valid_customer)
        assert response.status_code == 422
    
    def test_negative_charges(self, client, valid_customer):
        """Negative charges should return 422."""
        valid_customer["monthlycharges"] = -10.0
        response = client.post("/predict", json=valid_customer)
        assert response.status_code == 422
    
    def test_extra_field_rejected(self, client, valid_customer):
        """Extra fields should be rejected (extra='forbid')."""
        valid_customer["unknown_field"] = "value"
        response = client.post("/predict", json=valid_customer)
        assert response.status_code == 422


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_zero_tenure(self, client, valid_customer):
        """Zero tenure (brand new customer) should work."""
        valid_customer["tenure"] = 0
        valid_customer["totalcharges"] = 0.0
        response = client.post("/predict", json=valid_customer)
        assert response.status_code == 200
    
    def test_high_tenure(self, client, valid_customer):
        """Very high tenure should work."""
        valid_customer["tenure"] = 100
        valid_customer["totalcharges"] = 10000.0
        response = client.post("/predict", json=valid_customer)
        assert response.status_code == 200
    
    def test_senior_citizen(self, client, valid_customer):
        """Senior citizen flag should work."""
        valid_customer["seniorcitizen"] = 1
        response = client.post("/predict", json=valid_customer)
        assert response.status_code == 200
