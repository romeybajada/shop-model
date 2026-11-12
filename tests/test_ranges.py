"""Property-based test: the model must behave on every price the business will send,
including the premium range above anything in the training data."""
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from shop_model.config import data_path, load_config
from shop_model.data import load_data, training_window
from shop_model.features import add_vat
from shop_model.model import train_model

CONFIG = load_config()
# The most expensive item the shop will ever list, with headroom: a business number, not a sample statistic.
MAX_PRICE = 120.0


@pytest.fixture(scope="module")
def model():
    # The same rows the gate trains on, so this tests the model that ships.
    df = add_vat(training_window(load_data(data_path(CONFIG)), CONFIG), CONFIG["vat_rate"])
    return train_model(df, seed=CONFIG["seed"])


def prob(model, price):
    df = add_vat(pd.DataFrame({"price": [price]}), CONFIG["vat_rate"])
    return float(model.predict_proba(df[["price"]])[0][1])


@settings(max_examples=200, deadline=None)
@given(price=st.floats(min_value=1.0, max_value=MAX_PRICE))
def test_probability_is_a_probability(model, price):
    assert 0.0 <= prob(model, price) <= 1.0


@settings(max_examples=200, deadline=None)
@given(a=st.floats(min_value=1.0, max_value=MAX_PRICE), b=st.floats(min_value=1.0, max_value=MAX_PRICE))
def test_a_higher_price_never_sells_more(model, a, b):
    low, high = sorted((a, b))
    assert prob(model, high) <= prob(model, low) + 1e-9