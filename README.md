# Shop model

Predicts whether an item in the shop will sell, from its price.

## Requirements

Python 3.14 with the venv module, the same version as the Dockerfile. On Ubuntu:

```bash
sudo apt install python3.14 python3.14-venv
```
## Install

Run these from this folder, the one with pyproject.toml:

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 -m pip install -e ".[dev]"
cp env.example .env
```

## Test

```bash
pytest
```

## Train

```bash
shop-train
```

The model is saved to models/model.joblib. Shared settings are in
config/config.yaml; settings for your machine go in .env.

## Run the API

With the virtual environment active and the model trained (`shop-train`):

```bash
uvicorn shop_model.api:app --host 0.0.0.0 --port 8888
```

In a second terminal, check that it answers:

```bash
curl localhost:8888/health
curl -X POST localhost:8888/predict -H "Content-Type: application/json" -d '{"price": 20}'
```

Expected: `{"status":"ok"}` and `{"price":20.0,"probability_sold":0.675}`.
Interactive documentation is at http://localhost:8888/docs. Stop the server with Ctrl+C.

## Run in Docker

Needs Docker; no virtual environment required. The model is trained while the image is built.

```bash
docker build -t shop-model:0.3 .
docker run -d --name shop-api -p 8888:8888 shop-model:0.3
```

Check that it is running and answers:

```bash
docker ps
curl localhost:8888/health
curl -X POST localhost:8888/predict -H "Content-Type: application/json" -d '{"price": 20}'
docker logs shop-api
```

Stop and remove the container:

```bash
docker stop shop-api
docker rm shop-api
```

### Configuration

Settings are passed as environment variables, never baked into the image:

```bash
docker run -d --name shop-api -p 8888:8888 -e LOG_LEVEL=WARNING shop-model:0.3
docker run -d --name shop-api -p 8888:8888 --env-file .env shop-model:0.3
```

To serve a model trained outside the image, mount the models folder:

```bash
docker run -d --name shop-api -p 8888:8888 -v "$(pwd)/models:/app/models" shop-model:0.3
```

## Run with Docker Compose

Starts the API together with a PostgreSQL database that stores every prediction.
Needs Docker with the Compose plugin (`docker compose version`). Compose reads the
database password from `DB_PASSWORD` in .env, so create it first (`cp env.example .env`)
and set your own password.

From this folder, the one with compose.yaml:

```bash
docker compose up -d --build
docker compose ps
```

`db` should show as `(healthy)` and `api` as `Up`. Send a few requests:

```bash
curl -X POST localhost:8888/predict -H "Content-Type: application/json" -d '{"price": 8}'
curl -X POST localhost:8888/predict -H "Content-Type: application/json" -d '{"price": 20}'
curl -X POST localhost:8888/predict -H "Content-Type: application/json" -d '{"price": 35}'
```

Expected probabilities: 0.885, 0.675 and 0.288. Check that they were stored:

```bash
docker compose exec db psql -U shop -d shop -c "SELECT * FROM predictions;"
```

Each row holds the time, the price, the probability and the model version.

Useful commands:

| Command | What it does |
|---|---|
| `docker compose logs -f api` | Follow the API's log (Ctrl+C to stop) |
| `docker compose exec db psql -U shop -d shop` | Interactive psql session (`\q` to leave) |
| `docker compose stop api` / `start api` | Stop or start one service |
| `docker compose down` | Remove the containers; the stored predictions are kept |
| `docker compose down -v` | Also delete the database volume and its data |

The predictions table is created by db/init.sql, which only runs on an empty database.
After changing init.sql, run `docker compose down -v` and start again.

## Monitoring

`docker compose up -d --build` also starts Prometheus and Grafana (from
monitoring/compose.yaml). The API reports its metrics at `/metrics`:

```bash
curl localhost:8888/metrics
```

Open the **Model service** dashboard at http://localhost:3000 (no login needed).
Prometheus is at http://localhost:9090; Status, then Target health, shows whether
it can reach the API. On a VirtualBox VM, forward host port 3000 (and 9090 if wanted)
to the same guest port.

The panels are empty until requests arrive. Generate traffic, with the virtual
environment active:

```bash
python3 monitoring/traffic.py --field price --mean 23 --sd 12 --duration 300
```

This sends about 5 requests per second for 5 minutes, with prices around €23. On the
dashboard, requests per second rises to about 5, latency stays low, and the average
input value settles around 23 to 24.

To simulate drift (a premium range raising prices), run it with a higher mean:

```bash
python3 monitoring/traffic.py --field price --mean 35 --sd 12 --duration 300
```

The input panels shift while request rate and latency stay the same.
