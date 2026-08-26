FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY configs ./configs
COPY src ./src
COPY models ./models

EXPOSE 8000
CMD ["uvicorn", "sleep_mlops.api:app", "--host", "0.0.0.0", "--port", "8000"]
