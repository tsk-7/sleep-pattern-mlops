FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY configs ./configs
COPY datasets ./datasets
COPY src ./src
COPY data ./data
COPY models ./models

RUN mkdir -p /app/data/processed /app/models

EXPOSE 8000

CMD ["uvicorn", "sleep_mlops.api:app", "--host", "0.0.0.0", "--port", "8000"]
