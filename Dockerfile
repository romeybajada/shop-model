FROM python:3.14-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml .
COPY src ./src
COPY config ./config
COPY data ./data
RUN pip install --no-cache-dir --no-deps .

RUN shop-train

EXPOSE 8888
CMD ["uvicorn", "shop_model.api:app", "--host", "0.0.0.0", "--port", "8888"]
