FROM python:3.14-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml .
COPY src ./src
COPY config ./config
COPY data ./data
RUN pip install --no-cache-dir --no-deps .

# No tracking server exists while the image is built (Stage 15 removes this line).
RUN MLFLOW_TRACKING_URI=none shop-train

ARG MODEL_VERSION=dev
ENV MODEL_VERSION=${MODEL_VERSION}

EXPOSE 8888
CMD ["uvicorn", "shop_model.api:app", "--host", "0.0.0.0", "--port", "8888"]
