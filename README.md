# Shop model

Predicts whether an item in the shop will sell, from its price.

## Requirements

- Python 3.14 with the venv module, the same version as the Dockerfile and the CI workflow
- git and make
- Docker with the Compose plugin, for `make build`, `make up` and `make ci`

On Ubuntu:

```bash
sudo apt install python3.14 python3.14-venv git make
```

## Install

Run these from this folder, the one with pyproject.toml:

```bash
python3.14 -m venv .venv
source .venv/bin/activate
make install
cp env.example .env
```

`make install` installs the pinned packages in requirements.txt, then this package with
its dev tools (pytest, ruff, pandera, hypothesis). Edit .env for your machine and set
your own `DB_PASSWORD`. .env is never committed; env.example holds placeholders only.

## Data

| File | Rows | What it holds |
|---|---|---|
| data/prices_v2.csv | 560 | Dated listings from November 2025 to August 2026: description, price and whether it sold |
| data/holdout.csv | 120 | The eight weeks from 31 August 2026. Cut once, never edited, never trained on |

The holdout is frozen by its SHA-256 fingerprint, `holdout_sha256` in config/config.yaml.
Never edit it: if it changes, the gate refuses to run. .gitattributes stops git from
changing its line endings.

The model trains on the rows before `holdout_start`. `train_start` in config.yaml sets
where training begins: it is `2026-08-03`, the launch of the premium range, so the model
learns from the weeks after buyers' behaviour changed. Set it to `null` to train on
every row before the holdout.

## The pipeline

Every check runs from the Makefile, with the virtual environment active. `build` and
`up` also need Docker running. To run the whole pipeline in order:

```bash
make ci
```

`make ci` runs lint, test, validate, gate and build, and stops at the first one that fails.

| Target | What it does | Fails when |
|---|---|---|
| `make install` | Installs the pinned requirements and this package with its dev tools | A package cannot be installed |
| `make lint` | ruff checks src and tests for errors and bad style | A ruff rule is broken |
| `make test` | pytest runs every test, including the range and quality-check tests | Any test fails |
| `make validate` | Checks both data files against the pandera schema, and that the holdout is later than, and separate from, the training data | A row, column or the split is invalid |
| `make gate` | Trains on the rows before the holdout, then scores the frozen holdout | The gate metric is below the threshold in config.yaml |
| `make build` | Builds `shop-model:<version>` and `shop-model:<version>-<commit>`, the version read from pyproject.toml | The image does not build |
| `make up` | Starts the API, database and monitoring with Docker Compose, using that version | Compose fails |
| `make publish` | Pushes both image tags to the registry; run by CI on `main` only | The push is refused |
| `make version` | Prints the version | Never |
| `make ci` | Runs lint, test, validate, gate and build, in that order | Any of them fails |

### The quality gate

`make gate` prints every metric on the holdout, the accuracy of always predicting the
majority class, and a 95 % bootstrap interval for the gate metric. The gate metric and
threshold are `gate_metric` and `gate_threshold` in config.yaml: accuracy, at least 0.70.
The current model scores 0.767 (interval 0.683 to 0.842).

`shop-gate` exits with:

| Code | Meaning | Who acts |
|---|---|---|
| 0 | Pass | Nobody |
| 1 | The model is below the threshold on the holdout | The model's owner: retrain, or argue for a new threshold in writing |
| 2 | data/holdout.csv no longer matches its fingerprint | Whoever changed it: `git checkout data/holdout.csv` |
| 3 | A data file is missing | Check `DATA_PATH` in .env and the paths in config.yaml |

`make` reports these as `Error 1`, `Error 2` or `Error 3` on its last line.

To score the random 20 % split instead of the holdout, for comparison:

```bash
make gate GATE_SPLIT=random
```

### The version

The version lives only in pyproject.toml. `make build`, `make up` and `make publish` read
it from there and pass it into the image as `MODEL_VERSION`, which the API stores with
every prediction. To release a new version, see Continuous integration below.

## Continuous integration

GitHub Actions runs the pipeline on every push and every pull request, using
.github/workflows/ci.yml. Each run starts on a fresh Ubuntu machine, installs the
project with `make install`, copies env.example to .env (placeholders only, never real
secrets), and runs `make ci`: lint, test, validate, gate and build. A failing step turns
the `ci` check red. The `main` branch requires that check to pass before a pull
request can be merged.

On a push to `main`, and only if `make ci` passed, the workflow logs in to the GitHub
container registry with the run's own `GITHUB_TOKEN` and runs `make publish`. That
pushes the image `make ci` just built and checked, not a second build, to:

```
ghcr.io/romeybajada/shop-model:<version>
ghcr.io/romeybajada/shop-model:<version>-<commit>
```

The first tag is the version from pyproject.toml. The second adds the short commit hash,
so it always identifies the exact code the image was built from. Pull requests and
other branches run the checks but never publish. Published images are listed under
Packages on the GitHub profile.

To pull an image (if the package is private, first run `docker login ghcr.io` with a
personal access token (classic) that has the `read:packages` scope):

```bash
docker pull ghcr.io/romeybajada/shop-model:0.4.0
```

### Releasing a new version

1. Change `version` in pyproject.toml, for example to `0.4.1`.
2. Commit and push to `main`, or merge a pull request into it.
3. When the run passes, `ghcr.io/romeybajada/shop-model:0.4.1` is published. The image
   tag, the Docker build argument and the `MODEL_VERSION` the API reports all come from
   that one line.

If the gate or any other check fails, nothing is published.

## Test

```bash
make test
```

The tests cover the features and training (tests/test_shop_model.py), the validator,
split check and gate (tests/test_quality_checks.py), and two rules that must hold for
every price from €1 to €120 (tests/test_ranges.py): the probability stays between 0 and
1, and a higher price never makes a sale more likely.

## Train

```bash
shop-train
```

The model is trained on the training window (see Data) and saved to
models/model.joblib. Shared settings are in config/config.yaml; settings for your
machine go in .env, which overrides config.yaml.

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

Expected: `{"status":"ok"}` and `{"price":20.0,"probability_sold":0.975}`.
Interactive documentation is at http://localhost:8888/docs. Stop the server with Ctrl+C.

## Run in Docker

Needs Docker; no virtual environment required. The model is trained while the image is built.

```bash
make build
docker run -d --name shop-api -p 8888:8888 shop-model:0.4.0
```

`make build` tags the image with the version from pyproject.toml. Check that it is
running and answers:

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

To run the published image instead of building one, replace `shop-model:0.4.0` with
`ghcr.io/romeybajada/shop-model:0.4.0`.

### Configuration

Settings are passed as environment variables, never baked into the image:

```bash
docker run -d --name shop-api -p 8888:8888 -e LOG_LEVEL=WARNING shop-model:0.4.0
docker run -d --name shop-api -p 8888:8888 --env-file .env shop-model:0.4.0
```

To serve a model trained outside the image, mount the models folder:

```bash
docker run -d --name shop-api -p 8888:8888 -v "$(pwd)/models:/app/models" shop-model:0.4.0
```

## Run with Docker Compose

Starts the API together with a PostgreSQL database that stores every prediction.
Needs Docker with the Compose plugin (`docker compose version`). Compose reads the
database password from `DB_PASSWORD` in .env, so create it first (`cp env.example .env`)
and set your own password.

From this folder, the one with compose.yaml:

```bash
make up
docker compose ps
```

`make up` builds the image and starts the services with `MODEL_VERSION` set from
pyproject.toml. Plain `docker compose up -d --build` also works, but the version is
then `dev`. `db` should show as `(healthy)` and `api` as `Up`. Send a few requests:

```bash
curl -X POST localhost:8888/predict -H "Content-Type: application/json" -d '{"price": 8}'
curl -X POST localhost:8888/predict -H "Content-Type: application/json" -d '{"price": 20}'
curl -X POST localhost:8888/predict -H "Content-Type: application/json" -d '{"price": 35}'
```

Expected probabilities: 0.997, 0.975 and 0.747. Check that they were stored:

```bash
docker compose exec db psql -U shop -d shop -c "SELECT * FROM predictions;"
```

Each row holds the time, the price, the probability and the model version (`0.4.0`).

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

`make up` also starts Prometheus and Grafana (from monitoring/compose.yaml). The API
reports its metrics at `/metrics`:

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
