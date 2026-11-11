from prometheus_client import Counter, Histogram

PREDICTIONS = Counter(
    "predictions",
    "Number of predictions served",
)

LATENCY = Histogram(
    "prediction_latency_seconds",
    "Time taken to answer a prediction request",
)

INPUTS = Histogram(
    "prediction_input",
    "Input values received (here: price in euros)",
    buckets=[5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 80, 100],
)