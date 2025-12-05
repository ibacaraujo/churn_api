import pickle
from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel, ConfigDict

class Lead(BaseModel):
    model_config = ConfigDict(extra='forbid')

    lead_source: str
    number_of_courses_viewed: int
    annual_income: int

class PredictResponse(BaseModel):
    lead_probability: float
    lead: bool

app = FastAPI(title="lead-prediction")

with open('pipeline_v1.bin', 'rb') as f:
    pipeline = pickle.load(f)


def predict_single(lead):
    result = pipeline.predict_proba(lead)[0, 1]
    return float(result)

@app.post("/predict")
def predict(lead: Lead) -> PredictResponse:
    prob = predict_single(lead.model_dump())

    return PredictResponse(
        lead_probability=prob,
        lead=prob >= 0.5
    )



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)