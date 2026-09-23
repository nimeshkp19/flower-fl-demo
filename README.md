# Federated Learning Demo with Flower

A small, reproducible demonstration of **Federated Learning (FL)** using **Flower** and **PyTorch**.

The experiment simulates **10 clients** training a shared CNN on MNIST. The data is deliberately split in a **non-IID** way so that each client sees only a small subset of the digit classes. Clients train locally, return model parameters and metrics, and the server combines the client updates with **Federated Averaging (FedAvg)**.

> **Important:** This is a simulation. The 10 clients are virtual clients running on one machine; they are not 10 separate physical devices.

## Why this is interesting

- **Non-IID data:** the clients do not all see the same distribution of MNIST data. Each client is restricted to a small number of digit classes.
- **Local training:** each simulated client trains the current global CNN using only its own local partition.
- **Model aggregation:** clients return trained model parameters and metrics; the server combines the model updates with **FedAvg**, weighted by the amount of local training data.
- **Iterative learning:** the updated global model is sent to clients again for the next round.
- **Data locality:** in this simulation, the raw MNIST examples remain in the client-side partitions used for training; the federated workflow exchanges model information rather than uploading the raw training set to the server.

Federated Learning can reduce the need to centralize raw training data, but it is **not by itself a complete privacy guarantee**. Real deployments may require additional mechanisms such as secure aggregation, differential privacy, authentication, and encryption.

## Experiment configuration

| Setting | Value |
|---|---:|
| Dataset | MNIST |
| Model | Small CNN (PyTorch) |
| Simulated clients | 10 |
| Data distribution | Non-IID |
| Classes per client | 2 |
| Federated algorithm | FedAvg |
| Communication rounds | 5 |
| Local epochs | 1 |
| Batch size | 32 |
| Learning rate | 0.01 |
| Evaluation | Every round |

## Technology stack

This project uses several tools, each with a different role:

| Tool | Role in this project |
|---|---|
| **Flower (`flwr`)** | Federated Learning framework; coordinates the ServerApp/ClientApps and FedAvg workflow |
| **Ray** | Execution backend used by Flower's Simulation Runtime to run simulated clients |
| **PyTorch** | Defines the CNN and performs local training/evaluation |
| **Torchvision** | Vision/data utilities used with the PyTorch pipeline |
| **Flower Datasets (`flwr-datasets`)** | Creates and loads the federated MNIST partitions |
| **Matplotlib** | Visualizes the non-IID data distribution |

Flower's Simulation Runtime is built on Ray. Flower currently recommends **WSL2 for Windows users running simulations**, because Ray's native Windows support remains experimental. [Flower simulation documentation](https://flower.ai/docs/framework/how-to-run-simulations.html)

## Federated Learning frameworks

Flower is one framework in a broader Federated Learning ecosystem. Other well-known projects include:

- **TensorFlow Federated (TFF):** an open-source framework designed for federated learning and other computations on decentralized data, with APIs for both common FL workflows and custom federated algorithms. [TensorFlow Federated](https://www.tensorflow.org/federated)
- **NVIDIA FLARE:** an open-source, extensible FL SDK with local simulation, proof-of-concept, and production-oriented deployment workflows. [NVIDIA FLARE](https://nvflare.readthedocs.io/en/main/)
- **FedML:** a federated learning and distributed AI platform covering use cases such as smartphone/IoT, cross-silo, and browser-based FL. [FedML](https://open.fedml.ai/)

This project uses **Flower** because it provides a relatively compact way to demonstrate the client/server FL workflow while remaining compatible with a normal PyTorch training pipeline.

## Requirements

### Recommended environment for Windows users

- **Windows + WSL2 + Ubuntu**
- Python **3.12** is the environment used for this project
- Git
- Approximately **500 MB+** of free disk space, depending on the installed ML dependencies and downloaded data

Flower supports simulations directly on Windows, but its documentation notes that Ray support on Windows is experimental and recommends WSL2 for Windows simulation workloads. [Flower simulation documentation](https://flower.ai/docs/framework/how-to-run-simulations.html)

### Linux/macOS

A native Linux or macOS environment can be used for Flower simulations without the WSL2 layer.

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/nimeshkp19/flower-fl-demo.git
cd flower-fl-demo
```

### 2. Create and activate a virtual environment

Recommended in WSL2/Ubuntu:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install the project

```bash
python -m pip install -e .
```

The project's dependencies are declared in `pyproject.toml`.

## Run the demo

### 1. Visualize the non-IID split

Generate a chart showing which digit classes are assigned to the simulated clients:

```bash
python visualize_split.py
```

This is useful before training because it makes the non-IID assumption visible instead of treating the client partitions as a hidden implementation detail.

### 2. Run federated training

Configure the local simulation for 10 simulated clients:

```bash
flwr federation simulation-config --num-supernodes=10
```

Then start the Flower simulation and stream its logs:

```bash
flwr run . --stream
```

You should see the federated process progress through five rounds, including messages such as:

```text
Federation `@none/default` (10 simulated SuperNodes)

[ROUND 1/5]
configure_train: Sampled 10 nodes (out of 10)

aggregate_train: Received 10 results and 0 failures

[ROUND 2/5]
...

[ROUND 5/5]
...

Strategy execution finished in ...
```

The exact loss and accuracy values can vary between runs because local training and execution are not guaranteed to be numerically identical every time.

### 3. Optional: inspect the raw MNIST representation

```bash
python 00_look_at_data.py
```

This shows what an MNIST image looks like to a computer: a `1 × 28 × 28` tensor after conversion to a PyTorch tensor.

## Understanding one federated round

The basic workflow is:

```text
                 Global Model
                      |
          +-----------+-----------+
          |           |           |
          v           v           v
       Client 1    Client 2    ... Client 10
       local data  local data       local data
          |           |                |
       train        train            train
          |           |                |
          +-----------+----------------+
                      |
                      v
                 Model updates
                      |
                      v
                    FedAvg
                      |
                      v
                New global model
                      |
                   next round
```

A client starts from the current global model, trains locally on its own partition, and returns the updated model parameters and metrics. The server aggregates the client models using FedAvg and produces the next global model.

The same pattern repeats for each communication round.

## How the code is organized

```text
.
├── pyproject.toml          # Dependencies + Flower app configuration
├── visualize_split.py      # Visualizes the non-IID client split
├── 00_look_at_data.py      # Inspects a single MNIST image/tensor
└── fl_demo/
    ├── __init__.py
    ├── task.py              # CNN, data partitioning, training/evaluation
    ├── client_app.py        # Defines client-side train/evaluate tasks
    └── server_app.py        # Initializes the global model and FedAvg
```



## Configuration

The main run configuration is in `pyproject.toml` under `[tool.flwr.app.config]`:

| Key | Meaning |
|---|---|
| `num-server-rounds` | Number of federated training rounds |
| `fraction-evaluate` | Fraction of available clients used for evaluation |
| `learning-rate` | Learning rate used by each client's optimizer |
| `batch-size` | Local training batch size |
| `local-epochs` | Number of local epochs per federated round |

The number of simulated clients is controlled separately through the Flower federation configuration:

```bash
flwr federation simulation-config --num-supernodes=10
```

## Windows, WSL2, and practical issues

This project was originally tested directly on Windows and then moved to **WSL2/Ubuntu** for the Flower simulation workflow.

The main practical issue was the simulation backend: Flower's Simulation Runtime uses Ray, and Flower currently describes Ray's native Windows support as experimental while recommending WSL2 for Windows users running simulations. [Flower simulation documentation](https://flower.ai/docs/framework/how-to-run-simulations.html)

WSL2 provides a Linux environment inside Windows, allowing the project to keep using Windows while running the Flower/Ray simulation in Ubuntu.

### Runtime dependencies

Flower can create an isolated runtime environment for an app and install the dependencies declared in `pyproject.toml` automatically. This is convenient for reproducibility, but the first run can take significantly longer while dependencies are downloaded and installed. The managed local SuperLink can be configured to disable runtime dependency installation when dependencies have already been prepared locally. [Flower runtime dependency documentation](https://flower.ai/docs/framework/1.37/en/how-to-install-app-dependencies-at-runtime.html)

For a presentation, it is preferable to **run the experiment beforehand and use screenshots of the successful run**, rather than depending on package installation, dataset downloads, or local runtime services during the presentation itself.

## Useful Flower commands

Check the installed Flower version:

```bash
flwr --version
```

Run the simulation:

```bash
flwr run . --stream
```

List a completed or running simulation by ID:

```bash
flwr list --run-id <RUN_ID>
```

Show the logs for a run:

```bash
flwr log <RUN_ID> --show
```

## Presentation demo

This repository was created as the practical component of a short presentation on Federated Learning.

The recommended presentation demo is **screenshot-based** rather than a live execution. A successful `flwr run . --stream` output can be captured and explained step-by-step:

```text
10 simulated clients
        ↓
Round 1
        ↓
Local training
        ↓
10 client results returned
        ↓
FedAvg aggregation
        ↓
Evaluation
        ↓
Round 2 ... Round 5
```

The screenshots demonstrate the real experiment while avoiding presentation-time risks such as dependency installation, model downloads, Ray startup, or operating-system-specific runtime issues.

The repository can then be shared so others can reproduce the experiment themselves.

## Privacy note

Federated Learning changes **where training data is processed**, but it does not automatically make a system private or secure.

This demo illustrates the basic data-locality idea: the client partitions are used locally for training, while the federated workflow exchanges model information and metrics. Real systems may additionally use techniques such as secure aggregation, differential privacy, encryption, authentication, access control, and robust aggregation depending on the threat model and application.

## Background

The experiment is intended to demonstrate:

- centralized vs. federated training
- client/server architecture
- local model training
- non-IID data
- FedAvg aggregation
- simulation with Flower and Ray
- practical environment considerations such as WSL2 on Windows

## References and further reading

### Foundational paper

**McMahan et al. (2017)** — *Communication-Efficient Learning of Deep Networks from Decentralized Data*. This is the foundational paper associated with the Federated Averaging (FedAvg) approach used in this project.

- [Paper — Proceedings of Machine Learning Research (PMLR)](https://proceedings.mlr.press/v54/mcmahan17a)
- [Paper — arXiv](https://arxiv.org/abs/1602.05629)

### Federated Learning overview

**Yurdem et al. (2024)** — *Federated learning: Overview, strategies, applications, tools and future directions*. **Heliyon, 10(19), e38137.** This review provides a broad overview of federated learning, including FL strategies, applications, tools/frameworks, challenges, and future research directions.

- [Paper — Cell / Heliyon](https://www.cell.com/heliyon/fulltext/S2405-8440(24)14168-0)
- [Paper — ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2405844024141680)
- [Paper — PubMed](https://pubmed.ncbi.nlm.nih.gov/39391509/)
- DOI: [10.1016/j.heliyon.2024.e38137](https://doi.org/10.1016/j.heliyon.2024.e38137)

### Flower documentation

- [Flower Framework Documentation](https://flower.ai/docs/framework/) — main documentation
- [Flower Quickstart: PyTorch](https://flower.ai/docs/framework/tutorial-quickstart-pytorch.html) — build and run a Flower + PyTorch application
- [Run Flower Simulations](https://flower.ai/docs/framework/how-to-run-simulations.html) — simulation workflow and configuration
- [Flower Architecture](https://flower.ai/docs/framework/explanation-flower-architecture.html) — ServerApp, ClientApp, SuperLink, and federation architecture

These resources were used to understand the federated learning workflow, FedAvg, Flower application structure, and the simulation environment used by this demo.

## License

This project is licensed under the **MIT License**. See [`LICENSE`](LICENSE) for the full license text.
