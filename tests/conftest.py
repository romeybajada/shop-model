import numpy as np
import pandas as pd
import pytest

@pytest.fixture
def sample_df():
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "price": rng.uniform(5, 50, 50),
        "sold": rng.integers(0, 2, 50),
    })