install:
	uv sync

train:
	uv run train.py

run:
	uv run uvicorn predict:app --reload

test:
	uv run pytest test.py

lint:
	uv run ruff check .

load-test:
	uv run locust -f locustfile.py
