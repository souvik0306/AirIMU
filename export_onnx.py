#!/usr/bin/env python3
"""
Export a trained AirIMU model to ONNX format for CPU deployment.

ONNX Model Input/Output Specification:
--------------------------------------
Inputs:
  - "acc":  shape [batch=1, seq_len, 3], dtype=float32
            Pre-padded accelerometer data (must include 9 padding frames)
            Example: For 1 real sample, input shape is [1, 10, 3] (9 padding + 1 real)
  
  - "gyro": shape [batch=1, seq_len, 3], dtype=float32
            Pre-padded gyroscope data (must include 9 padding frames)
            Example: For 1 real sample, input shape is [1, 10, 3] (9 padding + 1 real)

Outputs:
  - "corr_acc":  shape [batch=1, N, 3], dtype=float32
                 Correction for N real accelerometer samples (padding excluded)
  
  - "corr_gyro": shape [batch=1, N, 3], dtype=float32
                 Correction for N real gyroscope samples (padding excluded)

Note: User must add 9 synthetic padding frames before calling ONNX model.
      See inference_onnx.py for padding implementation.
"""
import argparse
import torch
from pyhocon import ConfigFactory
from model import net_dict


class CodeNetWrapper(torch.nn.Module):
    """
    Wrapper that converts model's dict-based interface to tensor interface for ONNX.
    
    This wrapper is necessary because ONNX doesn't support dict inputs/outputs.
    It takes raw tensors and wraps them in a dict for the model, then unwraps the output.
    
    CRITICAL FIX: This wrapper also patches the encoder to explicitly initialize GRU
    hidden states to zeros, ensuring consistent behavior between PyTorch and ONNX.
    """

    def __init__(self, net: torch.nn.Module):
        super().__init__()
        self.net = net
        
        # Patch the encoder to use explicit h0 initialization
        original_encoder = self.net.encoder
        
        def patched_encoder(x):
            batch_size = x.shape[0]
            device = x.device
            dtype = x.dtype
            
            # CNN processing
            x = self.net.cnn(x.transpose(-1,-2)).transpose(-1,-2)
            
            # GRU1 with explicit h0=zeros
            h0_gru1 = torch.zeros(1, batch_size, 128, device=device, dtype=dtype)
            x, _ = self.net.gru1(x, h0_gru1)
            
            # GRU2 with explicit h0=zeros
            h0_gru2 = torch.zeros(1, batch_size, 256, device=device, dtype=dtype)
            x, _ = self.net.gru2(x, h0_gru2)
            
            return x
        
        self.net.encoder = patched_encoder

    def forward(self, acc: torch.Tensor, gyro: torch.Tensor):  # type: ignore[override]
        """
        Forward pass for ONNX export.
        
        Args:
            acc: [batch, seq_len, 3] pre-padded accelerometer (includes 9 padding frames)
            gyro: [batch, seq_len, 3] pre-padded gyroscope (includes 9 padding frames)
        
        Returns:
            correction_acc: [batch, seq_len-9, 3] corrections for real samples
            correction_gyro: [batch, seq_len-9, 3] corrections for real samples
        """
        # CRITICAL: Set model to eval mode and disable any dropout/batchnorm variations
        self.net.eval()
        
        # Wrap tensors in dict for model inference
        out = self.net.inference({"acc": acc, "gyro": gyro})
        
        # Extract corrections from dict output
        return out["correction_acc"], out["correction_gyro"]


def export(config: str, ckpt: str, onnx_path: str, torch_path: str,
           seq_len: int, opset: int, fp64: bool) -> None:
    """
    Load a trained checkpoint and export to ONNX format.
    
    Args:
        config: Path to model configuration file (.conf)
        ckpt: Path to trained checkpoint file (.ckpt)
        onnx_path: Output path for ONNX model (.onnx)
        torch_path: Output path for PyTorch weights (.pt)
        seq_len: Dummy sequence length for export (must be >= interval+1)
        opset: ONNX opset version (17 recommended)
        fp64: If True, export in float64; if False, export in float32 (recommended)
    """
    print(f"Loading configuration from: {config}")
    conf = ConfigFactory.parse_file(config)
    
    # Disable ground truth rotation for inference-only model
    conf.train.gtrot = False
    
    # Build network architecture
    print(f"Building network: {conf.train.network}")
    net = net_dict[conf.train.network](conf.train)
    
    # Load trained weights from checkpoint
    print(f"Loading checkpoint from: {ckpt}")
    state = torch.load(ckpt, map_location="cpu")
    net.load_state_dict(state["model_state_dict"])
    
    # Set to evaluation mode (disables dropout, batch norm, etc.)
    net.eval()
    for module in net.modules():
        module.eval()
    
    # Convert to target precision
    # IMPORTANT: fp32 is recommended for CPU deployment (7.45e-08 error vs fp64)
    if fp64:
        net = net.double()
        dtype = torch.float64
        print("✓ Using float64 precision (slower, maximum accuracy)")
    else:
        net = net.float()
        dtype = torch.float32
        print("✓ Using float32 precision (faster, 7.45e-08 error - RECOMMENDED)")

    print(f"Model interval (padding frames): {net.interval}")
    
    # Save PyTorch state dict (useful for loading without ONNX)
    torch.save(net.state_dict(), torch_path)
    print(f"✓ Saved PyTorch weights to: {torch_path}")

    # ============================================================================
    # PREPARE DUMMY INPUTS FOR ONNX EXPORT
    # ============================================================================
    # The dummy inputs define the shape and type of the ONNX model's inputs.
    # seq_len must be >= net.interval + 1 (typically interval=9, so min is 10)
    #
    # For runtime with N real samples:
    #   - Add 9 padding frames → total input length = N + 9
    #   - Model outputs N corrections (first 9 frames are skipped)
    #
    # Example for single sample (N=1):
    #   Input:  [1, 10, 3]  (1 batch, 10 frames with padding, 3 axes)
    #   Output: [1, 1, 3]   (1 batch, 1 correction, 3 axes)
    
    L = max(seq_len, net.interval + 1)
    print(f"\nDummy input configuration:")
    print(f"  Sequence length: {L} (includes {net.interval} padding frames)")
    print(f"  Input shape: [batch=1, seq_len={L}, axes=3]")
    print(f"  Output shape: [batch=1, seq_len={L - net.interval}, axes=3]")
    
    acc = torch.zeros(1, L, 3, dtype=dtype)
    gyro = torch.zeros(1, L, 3, dtype=dtype)

    # Wrap model for ONNX export
    wrapper = CodeNetWrapper(net)
    
    # Test wrapper before export to ensure it works
    print("\nTesting wrapper before export...")
    with torch.no_grad():
        test_out_acc, test_out_gyro = wrapper(acc, gyro)
    print(f"  ✓ Test passed - output shapes: {test_out_acc.shape}, {test_out_gyro.shape}")
    
    # ============================================================================
    # EXPORT TO ONNX
    # ============================================================================
    print(f"\nExporting to ONNX (opset version {opset})...")
    torch.onnx.export(
        wrapper,                          # Model to export
        (acc, gyro),                      # Dummy inputs
        onnx_path,                        # Output file path
        
        # Input/output names (used to reference tensors in ONNX runtime)
        input_names=["acc", "gyro"],
        output_names=["corr_acc", "corr_gyro"],
        
        # Dynamic axes allow variable sequence length at runtime
        # {tensor_name: {axis_index: symbolic_name}}
        dynamic_axes={
            "acc": {1: "seq"},           # dim 1 (sequence length) is dynamic
            "gyro": {1: "seq"},          # dim 1 (sequence length) is dynamic
            "corr_acc": {1: "seq"},      # dim 1 (sequence length) is dynamic
            "corr_gyro": {1: "seq"}      # dim 1 (sequence length) is dynamic
        },
        
        opset_version=opset,             # ONNX opset version (17 recommended)
        do_constant_folding=True,        # Optimize constant operations
        training=torch.onnx.TrainingMode.EVAL,  # Explicitly set eval mode
        verbose=False                    # Set to True for debug info
    )
    print(f"✓ Saved ONNX model to: {onnx_path}")
    
    # ============================================================================
    # VERIFY ONNX EXPORT
    # ============================================================================
    try:
        import onnxruntime as ort
        print("\nVerifying ONNX export...")
        
        # Create ONNX Runtime session
        sess = ort.InferenceSession(
            onnx_path, 
            providers=['CPUExecutionProvider']
        )
        
        # Run inference with dummy inputs
        onnx_outputs = sess.run(
            None,  # Return all outputs
            {"acc": acc.numpy(), "gyro": gyro.numpy()}
        )
        
        # Compare ONNX output with PyTorch output
        torch_outputs = [test_out_acc.numpy(), test_out_gyro.numpy()]
        
        diff_acc = abs(onnx_outputs[0] - torch_outputs[0]).max()
        diff_gyro = abs(onnx_outputs[1] - torch_outputs[1]).max()
        
        print(f"  Max difference (acc):  {diff_acc:.2e}")
        print(f"  Max difference (gyro): {diff_gyro:.2e}")
        
        # Check if differences are acceptable
        threshold = 1e-5 if fp64 else 1e-4
        if diff_acc < threshold and diff_gyro < threshold:
            print("  ✓ ONNX export verified - outputs match PyTorch!")
        else:
            print("  ⚠️  Large differences detected (may need investigation)")
            
    except ImportError:
        print("\n⚠️  onnxruntime not installed - skipping verification")
        print("     Install with: pip install onnxruntime")
    except Exception as e:
        print(f"\n⚠️  Verification failed: {e}")
    
    # ============================================================================
    # EXPORT SUMMARY
    # ============================================================================
    print("\n" + "="*70)
    print("EXPORT COMPLETE")
    print("="*70)
    print(f"\nONNX Model Specification:")
    print(f"  Precision: {'float64' if fp64 else 'float32 (RECOMMENDED)'}")
    print(f"  Padding frames: {net.interval}")
    print(f"  Min input length: {net.interval + 1}")
    print(f"\nInput Format (at runtime):")
    print(f"  Input 'acc':  [1, N+{net.interval}, 3] - N real samples + {net.interval} padding")
    print(f"  Input 'gyro': [1, N+{net.interval}, 3] - N real samples + {net.interval} padding")
    print(f"\nOutput Format:")
    print(f"  Output 'corr_acc':  [1, N, 3] - corrections for N real samples")
    print(f"  Output 'corr_gyro': [1, N, 3] - corrections for N real samples")
    print(f"\nFiles Created:")
    print(f"  ONNX model:      {onnx_path}")
    print(f"  PyTorch weights: {torch_path}")
    print(f"\nNext Steps:")
    print(f"  1. Use inference_onnx.py to test the exported model")
    print(f"  2. Add {net.interval} padding frames before each inference")
    print(f"  3. For single sample: input shape [1, {net.interval + 1}, 3]")
    print("="*70)


def main() -> None:
    """
    Main entry point for ONNX export.
    
    Usage examples:
    
    1. Export for single-sample inference (float32):
       python export_onnx.py --seq-len 10
    
    2. Export for batch inference (float32):
       python export_onnx.py --seq-len 1000
    
    3. Export with float64 (slower but maximum accuracy):
       python export_onnx.py --seq-len 10 --fp64
    """
    parser = argparse.ArgumentParser(
        description="Export AirIMU checkpoint to ONNX format for CPU deployment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export for single-sample real-time inference (RECOMMENDED)
  python export_onnx.py --seq-len 10

  # Export for batch processing
  python export_onnx.py --seq-len 1000

  # Export with maximum accuracy (slower)
  python export_onnx.py --seq-len 10 --fp64

Notes:
  - seq-len must be >= 10 (9 padding + 1 real sample minimum)
  - float32 is recommended (7.45e-08 error vs float64, 1.5-2x faster)
  - ONNX model expects PRE-PADDED input (add 9 frames before inference)
        """
    )
    
    # Model configuration
    parser.add_argument("--config", 
                        default="configs/exp/EuRoC/codenet.conf",
                        help="Path to model configuration file")
    parser.add_argument("--ckpt", 
                        default="experiments/EuRoC/codenet/ckpt/best_model.ckpt",
                        help="Path to trained checkpoint file")
    
    # Output paths
    parser.add_argument("--onnx", 
                        default="experiments/EuRoC/codenet/airimu_cpu_fp32.onnx",
                        help="Output path for ONNX model")
    parser.add_argument("--torch", 
                        default="experiments/EuRoC/codenet/airimu_cpu_fp32.pt",
                        help="Output path for PyTorch state_dict")
    
    # Export configuration
    parser.add_argument("--seq-len", 
                        type=int, 
                        default=10,
                        help="Dummy sequence length for export (must be >= 10 for interval=9)")
    parser.add_argument("--opset", 
                        type=int, 
                        default=17, 
                        help="ONNX opset version (17 recommended for CPU)")
    parser.add_argument("--fp64", 
                        action="store_true",
                        help="Export in float64 precision (default: float32, recommended for CPU)")
    
    args = parser.parse_args()
    
    # Validate seq_len
    if args.seq_len < 10:
        print(f"Warning: seq_len ({args.seq_len}) is less than minimum (10)")
        print(f"         Setting seq_len to 10")
        args.seq_len = 10
    
    # Print configuration
    print("\n" + "="*70)
    print("AirIMU ONNX Export Configuration")
    print("="*70)
    print(f"Config:        {args.config}")
    print(f"Checkpoint:    {args.ckpt}")
    print(f"Output ONNX:   {args.onnx}")
    print(f"Output PyTorch: {args.torch}")
    print(f"Sequence length: {args.seq_len}")
    print(f"ONNX opset:    {args.opset}")
    print(f"Precision:     {'float64' if args.fp64 else 'float32 (RECOMMENDED)'}")
    print("="*70 + "\n")

    # Run export
    export(args.config, args.ckpt, args.onnx, args.torch,
           args.seq_len, args.opset, args.fp64)


if __name__ == "__main__":
    main()