FROM python:3.11-slim

ARG MODEL_NAME=unknown
ARG MODEL_VERSION=unknown
ARG MODEL_RUN_ID=unknown
ARG MODEL_BUNDLE_SHA256=unknown

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

LABEL org.opencontainers.image.title="Churn Prediction API" \
      com.churn.model-name="${MODEL_NAME}" \
      com.churn.model-version="${MODEL_VERSION}" \
      com.churn.model-run-id="${MODEL_RUN_ID}" \
      com.churn.bundle-sha256="${MODEL_BUNDLE_SHA256}"

COPY requirements-api.txt .

RUN pip install --no-cache-dir -r requirements-api.txt

COPY api/ ./api/
COPY deployment/ ./deployment/
COPY config.yaml .

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
