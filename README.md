# Customer Churn Prediction Service

This project implements a machine learning service for predicting customer churn. It exposes a REST API using FastAPI and serves a Logistic Regression model trained on the Telco Customer Churn dataset.

## Features

- **Model Training**: Automated pipeline to download data, preprocess, and train a Logistic Regression model.
- **Prediction API**: Fast and validated API endpoints using FastAPI and Pydantic.
- **Containerization**: Docker support for easy deployment, utilizing `uv` for fast package management.
- **Observability**: Real-time monitoring with Prometheus and Grafana.
- **Experiment Tracking**: Model versioning and metric tracking with MLflow.

## Prerequisites

- Python 3.10 or higher
- [uv](https://github.com/astral-sh/uv) (recommended for dependency management)
- Docker (optional, for containerized run)

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd deployment
   ```

2. **Install dependencies:**
   Using `uv`:
   ```bash
   uv sync
   ```
   Or using pip:
   ```bash
   pip install -r pyproject.toml
   ```

## Usage

### 1. Train the Model

Before running the service, you need to train the model and save it as `model.bin`.

```bash
uv run train.py
# or
python train.py
```

This script will:
- Download the Telco Customer Churn dataset.
- Preprocess the data (handle categorical variables, missing values).
- Train a Logistic Regression model.
- Save the trained pipeline to `model.bin`.

### 2. Run the API Server

Start the FastAPI server:

```bash
uv run uvicorn predict:app --reload
# or
uvicorn predict:app --reload
```

The API will be available at `http://localhost:8000`.

- **API Documentation**: Visit `http://localhost:8000/docs` for the interactive Swagger UI.

### 3. Make Predictions

You can test the API using the provided `test.py` script:

```bash
uv run test.py
# or
python test.py
```

Or use `curl`:

```bash
curl -X 'POST' \
  'http://localhost:8000/predict' \
  -H 'Content-Type: application/json' \
  -d '{
  "gender": "female",
  "seniorcitizen": 0,
  "partner": "yes",
  "dependents": "no",
  "phoneservice": "no",
  "multiplelines": "no_phone_service",
  "internetservice": "dsl",
  "onlinesecurity": "no",
  "onlinebackup": "yes",
  "deviceprotection": "no",
  "techsupport": "no",
  "streamingtv": "no",
  "streamingmovies": "no",
  "contract": "month-to-month",
  "paperlessbilling": "yes",
  "paymentmethod": "electronic_check",
  "tenure": 1,
  "monthlycharges": 29.85,
  "totalcharges": 29.85
}'
```

## Docker

### Build the Image

```bash
docker build -t churn-prediction .
```

### Run the Container

```bash
docker run -p 8000:8000 churn-prediction
```

The service is now running inside a container and accessible at `http://localhost:8000`.

## MLOps Stack

To run the full stack with Monitoring and Experiment Tracking:

```bash
docker-compose up --build
```

Services available:
- **API**: `http://localhost:8000`
- **Prometheus**: `http://localhost:9090`
- **Grafana**: `http://localhost:3000` (User: `admin`, Pass: `admin`)
- **MLflow**: `http://localhost:5000`

## Deployment (Fly.io)

This project is configured for continuous deployment to [Fly.io](https://fly.io).

### Setup

1.  **Install Fly CLI**:
    ```bash
    curl -L https://fly.io/install.sh | sh
    ```
2.  **Login**:
    ```bash
    fly auth login
    ```
3.  **Launch App** (First time only):
    ```bash
    fly launch
    ```
    - Follow the prompts.
    - When asked to deploy now, say **No** (we want GitHub to do it).

4.  **Configure Secrets**:
    - Get a token: `fly tokens create deploy -x 999999h`
    - Go to your GitHub Repo -> Settings -> Secrets -> Actions.
    - Add a new secret named `FLY_API_TOKEN` with the token value.

### Deploy

Just push to the `main` branch!
```bash
git push origin main
```
The GitHub Action will automatically build and deploy your app.
