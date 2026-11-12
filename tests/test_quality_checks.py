"""The checks that guard the pipeline must themselves be tested."""
import pandas as pd
import pandera.pandas as pa
import pytest

from shop_model import evaluate
from shop_model.validate import check_split, validate_file

GOOD = {"listed_at": ["2026-01-05"], "description": ["Herb garden kit"], "price": [19.87], "sold": [1]}


def test_validate_accepts_a_good_row(tmp_path):
    path = tmp_path / "good.csv"
    pd.DataFrame(GOOD).to_csv(path, index=False)
    assert len(validate_file(path)) == 1


@pytest.mark.parametrize("column,value", [("price", -5.0), ("sold", 2), ("listed_at", "not a date")])
def test_validate_rejects_a_bad_row(tmp_path, column, value):
    path = tmp_path / "bad.csv"
    pd.DataFrame({**GOOD, column: [value]}).to_csv(path, index=False)
    with pytest.raises(pa.errors.SchemaErrors):
        validate_file(path)


def test_validate_rejects_an_unexpected_column(tmp_path):
    path = tmp_path / "extra.csv"
    pd.DataFrame({**GOOD, "discount": [0.1]}).to_csv(path, index=False)
    with pytest.raises(pa.errors.SchemaErrors):
        validate_file(path)


def frame(rows):
    df = pd.DataFrame(rows, columns=["listed_at", "description", "price", "sold"])
    df["listed_at"] = pd.to_datetime(df["listed_at"])
    return df


def test_split_check_passes_a_clean_split_and_names_a_broken_one():
    data = frame([["2026-08-01", "Herb garden kit", 19.87, 1]])
    assert check_split(data, frame([["2026-09-01", "Glass carafe", 26.92, 0]]), "2026-08-31") == []
    early = check_split(data, frame([["2026-08-20", "Glass carafe", 26.92, 0]]), "2026-08-31")
    assert "before holdout_start" in early[0]
    shared = check_split(data, frame([["2026-08-01", "Herb garden kit", 19.87, 1]]), "2026-07-01")
    assert "both" in shared[0]


def test_gate_refuses_a_changed_holdout(tmp_path, monkeypatch):
    path = tmp_path / "holdout.csv"
    pd.DataFrame(GOOD).to_csv(path, index=False)
    config = evaluate.load_config()
    config["holdout_path"] = str(path)  # same SHA-256 expected, different file
    monkeypatch.setattr(evaluate, "load_config", lambda: config)
    with pytest.raises(SystemExit) as exit_info:
        evaluate.main([])
    assert exit_info.value.code == 2
