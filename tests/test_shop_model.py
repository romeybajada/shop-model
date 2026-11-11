import pandas as pd
import pytest

from shop_model.features import add_vat
from shop_model.model import train_model

def test_add_vat_value():
    df = pd.DataFrame({"price": [100.0]})
    result = add_vat(df, 0.18)
    assert result["price"].iloc[0] == pytest.approx(118.0)

def test_add_vat_does_not_modify_input(sample_df):
    original = sample_df.copy()
    add_vat(sample_df, 0.18)
    pd.testing.assert_frame_equal(sample_df, original)

def test_training_smoke_and_determinism(sample_df):
    a = train_model(sample_df, seed=42).predict_proba(sample_df[["price"]])[:, 1]
    b = train_model(sample_df, seed=42).predict_proba(sample_df[["price"]])[:, 1]
    assert len(a) == len(sample_df)
    assert ((a >= 0) & (a <= 1)).all()
    assert a == pytest.approx(b)

def test_add_vat_zero_rate_changes_nothing(sample_df):
    result = add_vat(sample_df, 0.0)
    pd.testing.assert_frame_equal(result, sample_df)