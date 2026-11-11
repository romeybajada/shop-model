def add_vat(df, rate):
    out = df.copy()
    out["price"] = out["price"] * (1 + rate)
    return out