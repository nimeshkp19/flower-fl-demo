"""fl_demo: ServerApp - coordinates FedAvg across the simulated clients."""

import torch

from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from fl_demo.task import Net

app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Entry point for the ServerApp: initialize the model and run FedAvg."""

    num_rounds: int = context.run_config["num-server-rounds"]
    fraction_evaluate: float = context.run_config["fraction-evaluate"]
    lr: float = context.run_config["learning-rate"]

    # Initial global model, sent to clients in round 1
    global_model = Net()
    arrays = ArrayRecord(global_model.state_dict())

    strategy = FedAvg(fraction_evaluate=fraction_evaluate)

    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
    )

    print("\nSaving final model to disk...")
    torch.save(result.arrays.to_torch_state_dict(), "final_model.pt")
