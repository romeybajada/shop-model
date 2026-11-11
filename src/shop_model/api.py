import logging
import os

import joblib
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel

from shop_model.config import load_config
from shop_model.db import save_prediction
from shop_model.features import add_vat
from shop_model.logging_setup import setup_logging
from shop_model.metrics import INPUTS, LATENCY, PREDICTIONS

load_dotenv()
setup_logging(os.environ.get("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

config = load_config()
MODEL_PATH = os.environ.get("MODEL_PATH", config["model_path"])
MODEL_VERSION = os.environ.get("MODEL_VERSION", "dev")
model = joblib.load(MODEL_PATH)
logger.info("Model loaded from %s", MODEL_PATH)

app = FastAPI(title="Shop sales model")

class Item(BaseModel):
    price: float

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/predict")
def predict(item: Item):
    with LATENCY.time():
        df = add_vat(pd.DataFrame({"price": [item.price]}), config["vat_rate"])
        probability = float(model.predict_proba(df[["price"]])[0][1])
        logger.info("price=%.2f probability=%.3f", item.price, probability)
        save_prediction(item.price, probability, MODEL_VERSION)
    PREDICTIONS.inc()
    INPUTS.observe(item.price)
    return {"price": item.price, "probability_sold": round(probability, 3)}