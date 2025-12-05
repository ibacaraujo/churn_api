from locust import HttpUser, task, between

class ChurnUser(HttpUser):
    wait_time = between(1, 5)

    @task
    def predict(self):
        self.client.post("/predict", json={
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
        })
