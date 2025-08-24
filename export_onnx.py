#!/usr/bin/env python3
"""Export a trained AirIMU model to ONNX and plain PyTorch weights."""
import argparse
import torch
from pyhocon import ConfigFactory
from model import net_dict


class CodeNetWrapper(torch.nn.Module):
    """Wrapper removing dict inputs for ONNX export."""

    def __init__(self, net: torch.nn.Module):
        super().__init__()
        self.net = net

    def forward(self, acc: torch.Tensor, gyro: torch.Tensor):  # type: ignore[override]
        out = self.net.inference({"acc": acc, "gyro": gyro})
        return out["correction_acc"], out["correction_gyro"]


def export(config: str, ckpt: str, onnx_path: str, torch_path: str,
           seq_len: int, opset: int, fp64: bool) -> None:
    """Load a checkpoint and export to ONNX and PyTorch formats."""
    conf = ConfigFactory.parse_file(config)
    conf.train.gtrot = False
    net = net_dict[conf.train.network](conf.train).eval()

    # Cast network to desired precision (float32 by default)
    if fp64:
        net = net.double()
        dtype = torch.float64
    else:
        net = net.float()
        dtype = torch.float32

    state = torch.load(ckpt, map_location="cpu")
    net.load_state_dict(state["model_state_dict"])

    # Save plain PyTorch weights
    torch.save(net.state_dict(), torch_path)

    # Prepare dummy inputs (seq_len >= net.interval + 1)
    L = max(seq_len, net.interval + 1)
    acc = torch.zeros(1, L, 3, dtype=dtype)
    gyro = torch.zeros(1, L, 3, dtype=dtype)

    wrapper = CodeNetWrapper(net)
    torch.onnx.export(
        wrapper,
        (acc, gyro),
        onnx_path,
        input_names=["acc", "gyro"],
        output_names=["corr_acc", "corr_gyro"],
        dynamic_axes={"acc": {1: "seq"}, "gyro": {1: "seq"},
                      "corr_acc": {1: "seq"}, "corr_gyro": {1: "seq"}},
        opset_version=opset,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Export AirIMU checkpoint to ONNX")
    parser.add_argument("--config", default="configs/exp/EuRoC/codenet.conf",
                        help="Path to model config file")
    parser.add_argument("--ckpt", default="experiments/EuRoC/codenet/ckpt/best_model.ckpt",
                        help="Checkpoint file")
    parser.add_argument("--onnx", default="codenet.onnx",
                        help="Output ONNX file")
    parser.add_argument("--torch", default="codenet.pt",
                        help="Output PyTorch state_dict file")
    parser.add_argument("--seq-len", type=int, default=1000,
                        help="Dummy sequence length for export (>= interval+1)")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    parser.add_argument("--fp64", action="store_true",
                        help="Export in float64 precision (default: float32)")
    args = parser.parse_args()

    export(args.config, args.ckpt, args.onnx, args.torch,
           args.seq_len, args.opset, args.fp64)


if __name__ == "__main__":
    main()
