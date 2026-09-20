# Federated Learning Demo with Flower

A small, runnable demo of **Federated Learning**: multiple clients
train a shared model collaboratively, without ever sending their
raw data anywhere.

Built with [Flower](https://flower.ai/) (`flwr`) and PyTorch, using
MNIST split in a deliberately **non-IID** way across simulated
clients, so you can watch a single global model learn to recognize
all 10 digits even though no individual client ever sees more than
a couple of them.

## Why this is interesting

- Each of the 5 simulated clients only holds **2 of the 10 digit
  classes** (see `visualize_split.py`) — on its own, no client
  could ever train a full digit classifier.
- Clients train locally and send back only their **model weights**
  — never their images.
- The server combines those weights with **Federated Averaging
  (FedAvg)**, weighted by how much data each client had.
- After a few rounds, the *shared* model classifies all 10 digits
  well, despite no client individually knowing more than 2.

## Requirements

- Python **3.11–3.13** (Flower's simulation backend, Ray, may not
  yet fully support brand-new Python releases like 3.14 — stick to
  3.11–3.13 for a smooth install)
- ~500MB free disk space (PyTorch + MNIST)

## Setup

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e .
```

## Run the demo

**1. See the non-IID split** — generates a chart showing exactly
which digits each client owns:

```bash
python visualize_split.py
```

**2. Run federated training** — 5 simulated clients, 5 rounds of
FedAvg, logged live to your terminal:

```bash
flwr run . --federation-config="num-supernodes=5" --stream
```

Watch `eval_acc` climb round over round in the log output. That's
the global model improving, aggregated from clients that individually
only ever saw a fraction of the problem.

**Optional — look at the raw data first:**

```bash
python 00_look_at_data.py
```
A quick look at what an MNIST image actually is under the hood
(a 28x28 grid of numbers) before any training happens.

## Project structure

```
.
├── pyproject.toml          # dependencies + Flower app config
├── visualize_split.py      # shows the non-IID split across clients
├── 00_look_at_data.py      # raw look at a single MNIST image
└── fl_demo/
    ├── task.py             # model, data partitioning, train/test loops
    ├── client_app.py       # runs on each simulated client
    └── server_app.py       # coordinates FedAvg on the server
```

## How it works, briefly

1. **`task.py`** partitions MNIST across clients using Flower
   Datasets' `PathologicalPartitioner`, so each client only gets 2
   digit classes — a deliberately hard, non-IID split.
2. **`client_app.py`** defines what a client does when asked to
   train or evaluate: load the current global model, train/test on
   its own local partition only, send back weights + metrics.
3. **`server_app.py`** initializes a model and runs Flower's
   `FedAvg` strategy for a configurable number of rounds,
   aggregating client updates each round.
4. `flwr run .` launches Flower's Simulation Engine, which
   simulates all of this locally, no real network required.

## Configuration

Tweak `[tool.flwr.app.config]` in `pyproject.toml`:

| Key | Meaning |
|---|---|
| `num-server-rounds` | how many rounds of federated training to run |
| `fraction-evaluate` | fraction of clients used for evaluation each round |
| `learning-rate` | local optimizer learning rate |
| `batch-size` | local training batch size |
| `local-epochs` | how many local epochs each client trains per round |

## Background

This demo was built as the practical component of a short
presentation on Federated Learning, covering the client-server
architecture, FedAvg, non-IID data, and the privacy properties (and
limits) of the approach.

## License

MIT — see [LICENSE](LICENSE).
