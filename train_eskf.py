"""Fine-tune only AirIMU covariance heads through full-flight ESKF replay."""

import argparse
import json
import math
from pathlib import Path
import random

import torch

from ekf import DifferentiableESKF
from ekf.data import prepare_flight, read_cutoff_table, read_flight_manifest
from ekf.experiment import load_airimu, save_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Existing AirIMU network HOCON config")
    parser.add_argument("--ckpt", required=True, help="AirIMU checkpoint (covariance heads are retained)")
    parser.add_argument("--flights", required=True, help="JSON manifest with train/validation flights")
    parser.add_argument("--cutoffs", required=True, help="CSV table of original-reference cutoff seconds")
    parser.add_argument("--output", required=True, help="Experiment directory for ESKF checkpoints")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--gravity", type=float, nargs=3, default=(0., 0., -9.80665))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cpu-threads", type=int, default=1)
    args = parser.parse_args()
    if args.epochs < 1 or not math.isfinite(args.lr) or args.lr <= 0 or args.cpu_threads < 1:
        parser.error("epochs, lr, and cpu-threads must be positive.")
    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    flights = read_flight_manifest(args.flights)
    cutoffs = read_cutoff_table(args.cutoffs)
    training = [record for record in flights if record["split"] == "train"]
    validation = [record for record in flights if record["split"] == "validation"]
    if not training or not validation:
        parser.error("Provide disjoint train and validation flights in the manifest.")
    for record in training + validation:
        if record["id"] not in cutoffs:
            parser.error(f"No cutoff table entry for {record['id']}.")
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    ekf = DifferentiableESKF(gravity=args.gravity).to(args.device)
    wrapper = load_airimu(args.config, args.ckpt, ekf, args.device)
    parameters = [p for p in wrapper.network.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=args.lr)
    metadata = {"gravity": list(args.gravity), "observation_types": list(ekf.observation_types),
                "timestamp_units": "seconds, original shared reference",
                "evaluation_start_offset": 5., "evaluation_duration": 20.,
                "config": str(Path(args.config).resolve()),
                "initial_checkpoint": str(Path(args.ckpt).resolve()),
                "flights": [{**record, "cutoff_timestamp": cutoffs[record["id"]]}
                            for record in training + validation], "seed": args.seed}
    with (output_dir / "eskf_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    best_validation = math.inf
    for epoch in range(args.epochs):
        wrapper.train()
        rng.shuffle(training)
        train_losses = []
        for record in training:
            optimizer.zero_grad(set_to_none=True)
            flight = prepare_flight(record, cutoffs[record["id"]], ekf)
            output = wrapper(**flight)
            loss = output["loss"]
            if not bool(torch.isfinite(loss)) or not loss.requires_grad:
                raise RuntimeError(f"{record['id']}: no finite differentiable velocity loss.")
            loss.backward()
            if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in parameters):
                raise RuntimeError(f"{record['id']}: non-finite covariance-head gradient.")
            optimizer.step()
            value = float(loss.detach())
            train_losses.append(value)
            print(f"epoch={epoch+1} flight={record['id']} train_xy_mse={value:.9g}", flush=True)
            # Release this flight's graph before starting the next complete replay.
            del loss, output, flight
        wrapper.eval()
        validation_losses = []
        # Evaluation has its own no-grad scope; training replay never enters it.
        with torch.no_grad():
            for record in validation:
                flight = prepare_flight(record, cutoffs[record["id"]], ekf)
                output = wrapper(**flight)
                validation_losses.append(float(output["loss"]))
                del output, flight
        validation_mse = sum(validation_losses)/len(validation_losses)
        if not math.isfinite(validation_mse):
            raise RuntimeError("Validation loss is non-finite.")
        if validation_mse < best_validation:
            best_validation = validation_mse
            save_checkpoint(output_dir / "best_model.ckpt", wrapper, optimizer, epoch+1,
                            best_validation, metadata)
        save_checkpoint(output_dir / "newest.ckpt", wrapper, optimizer, epoch+1, best_validation, metadata)
        record = {"epoch": epoch+1, "train_xy_mse": sum(train_losses)/len(train_losses),
                  "validation_xy_mse": validation_mse, "best_validation_xy_mse": best_validation}
        with (output_dir / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
