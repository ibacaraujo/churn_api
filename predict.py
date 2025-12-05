import pickle
import uvicorn
import pandas as pd
from pandera.errors import SchemaError
from fastapi import FastAPI, HTTPException
from typing import Dict, Any
from typing import Literal
from pydantic import BaseModel, Field, ConfigDict
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Histogram, Counter
from schemas import InputSchema

class Customer(BaseModel):
    model_config = ConfigDict(extra='forbid')

    gender: Literal["male", "female"]
    seniorcitizen: Literal[0, 1]
    partner: Literal["yes", "no"]
    dependents: Literal["yes", "no"]
    phoneservice: Literal["yes", "no"]
    multiplelines: Literal["yes", "no", "no_phone_service"]
    internetservice: Literal["dsl", "fiber_optic", "no"]
    onlinesecurity: Literal["no", "yes", "no_internet_service"]
    onlinebackup: Literal["no", "yes", "no_internet_service"]
    deviceprotection: Literal["no", "yes", "no_internet_service"]
    techsupport: Literal["no", "yes", "no_internet_service"]
    streamingtv: Literal["no", "yes", "no_internet_service"]
    streamingmovies: Literal["no", "yes", "no_internet_service"]
    contract: Literal["month-to-month", "one_year", "two_year"]
    paperlessbilling: Literal["yes", "no"]
    paymentmethod: Literal[
        "electronic_check",
        "mailed_check",
        "bank_transfer_(automatic)",
        "credit_card_(automatic)",
    ]
    tenure: int = Field(..., ge=0)
    monthlycharges: float = Field(..., ge=0.0)
    totalcharges: float = Field(..., ge=0.0)

class PredictResponse(BaseModel):
    churn_probability: float
    churn: bool

app = FastAPI(title="customer-churn-prediction")

# --- Prometheus Instrumentation ---
instrumentator = Instrumentator()
instrumentator.instrument(app).expose(app)

PREDICTION_VALUE = Histogram(
    "prediction_value",
    "Histogram of predicted churn probabilities",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
)

CHURN_COUNT = Counter(
    "churn_prediction_count",
    "Count of churn vs non-churn predictions",
    ["churn_status"]
)
# ----------------------------------

with open('model.bin', 'rb') as f:
    pipeline = pickle.load(f)


def predict_single(customer):
    result = pipeline.predict_proba(customer)[0, 1]
    
    # Update metrics
    PREDICTION_VALUE.observe(result)
    churn_status = "churn" if result >= 0.5 else "no_churn"
    CHURN_COUNT.labels(churn_status=churn_status).inc()
    
    return float(result)

@app.post("/predict")
def predict(customer: Customer) -> PredictResponse:
    customer_dict = customer.model_dump()
    
    # --- Data Validation (Pandera) ---
    customer_df = pd.DataFrame([customer_dict])
    try:
        InputSchema.validate(customer_df)
    except SchemaError as e:
        raise HTTPException(status_code=400, detail=str(e))
    # ---------------------------------

    prob = predict_single(customer_dict)

    return PredictResponse(
        churn_probability=prob,
        churn=prob >= 0.5
    )



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)