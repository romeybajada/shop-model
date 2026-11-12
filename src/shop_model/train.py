import logging
import os
import sys

from dotenv import load_dotenv

from shop_model.config import load_config, data_path
from shop_model.data import load_data, training_window
from shop_model.features import add_vat
from shop_model.logging_setup import setup_logging
from shop_model.model import save_model, train_model
from shop_model.seed import set_seed

logger = logging.getLogger(__name__)

def main():
    load_dotenv()
    config = load_config()
    set_seed(config["seed"])
    setup_logging(os.environ.get("LOG_LEVEL", "INFO"))

    try:
        DATA_PATH = data_path (config)
        MODEL_PATH = os.environ.get ("MODEL_PATH", config["model_path"])

        df1 = load_data (DATA_PATH)
        logger.info ("Data read from %s", DATA_PATH)

        df1 = training_window (df1, config)
        logger.info ("Training on %d rows in the training window", len(df1))

        df2 = add_vat (df1, config["vat_rate"])
        logger.info ("VAT rate of %.2f added", config["vat_rate"])

        model = train_model (df2, seed=config["seed"])
        logger.info ("Model trained using seed %d", config["seed"])

        save_model (model, MODEL_PATH)
        logger.info ("Model saved in %s", MODEL_PATH)
    except Exception:
        logger.exception ("Training failed")
        sys.exit (1)

if __name__ == "__main__":
    main()