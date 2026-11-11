import logging
from pathlib import Path

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

FEATURES = ["price"]

def train_model(df, seed):
    X_train, X_test, y_train, y_test = train_test_split(df[FEATURES], df["sold"], test_size=0.2, random_state=seed)
    model = LogisticRegression(random_state=seed)
    model.fit(X_train, y_train)
    logger.info("Test accuracy: %.3f", model.score(X_test, y_test))
    return model

def save_model(model, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)