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
  
  - "cov_acc":   shape [batch=1, N, 3], dtype=float32
                 Covariance for N real accelerometer samples (zeros if disabled)
  
  - "cov_gyro":  shape [batch=1, N, 3], dtype=float32
                 Covariance for N real gyroscope samples (zeros if disabled)

Note: User must add 9 synthetic padding frames before calling ONNX model.
      See inference_onnx.py for padding implementation.
      Covariance outputs are zeros if model was not trained with propcov=True.
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
            cov_acc: [batch, seq_len-9, 3] covariance for acc (zeros if disabled)
            cov_gyro: [batch, seq_len-9, 3] covariance for gyro (zeros if disabled)
        """
        # CRITICAL: Set model to eval mode and disable any dropout/batchnorm variations
        self.net.eval()
        
        # Wrap tensors in dict for model inference
        out = self.net.inference({"acc": acc, "gyro": gyro})
        
        # Extract corrections from dict output
        corr_acc = out["correction_acc"]
        corr_gyro = out["correction_gyro"]
        
        # Extract covariance (use zeros if not available)
        cov_state = out.get("cov_state", {})
        cov_acc = cov_state.get("acc_cov")
        cov_gyro = cov_state.get("gyro_cov")
        
        # If covariance is None, return zeros with same shape as corrections
        if cov_acc is None:
            cov_acc = torch.zeros_like(corr_acc)
        if cov_gyro is None:
            cov_gyro = torch.zeros_like(corr_gyro)
        
        return corr_acc, corr_gyro, cov_acc, cov_gyro


def export(config: str, ckpt: str, onnx_path: str, torch_path: str,
           opset: int, num_imu_frames: int) -> None:
    """
    Load a trained checkpoint and export to ONNX format with FIXED input shape.
    
    Args:
        config: Path to model configuration file (.conf)
        ckpt: Path to trained checkpoint file (.ckpt)
        onnx_path: Output path for ONNX model (.onnx)
        torch_path: Output path for PyTorch weights (.pt)
        opset: ONNX opset version (17 recommended)
        num_imu_frames: Number of real IMU frames to process (padding added automatically)
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
    
    # Always use float32 precision
    net = net.float()
    dtype = torch.float32
    print("✓ Using float32 precision")

    padding_frames = net.interval
    print(f"Model padding frames: {padding_frames}")
    
    # Save PyTorch state dict (useful for loading without ONNX)
    torch.save(net.state_dict(), torch_path)
    print(f"✓ Saved PyTorch weights to: {torch_path}")

    # ============================================================================
    # PREPARE DUMMY INPUTS FOR ONNX EXPORT
    # ============================================================================
    total_seq_len = padding_frames + num_imu_frames
    print(f"\nInput configuration:")
    print(f"  Real IMU frames: {num_imu_frames}")
    print(f"  Padding frames:  {padding_frames}")
    print(f"  Total input:     [1, {total_seq_len}, 3]")
    print(f"  Output:          [1, {num_imu_frames}, 3]")
    
    acc = torch.zeros(1, total_seq_len, 3, dtype=dtype)
    gyro = torch.zeros(1, total_seq_len, 3, dtype=dtype)

    # Wrap model for ONNX export
    wrapper = CodeNetWrapper(net)
    
    # Test wrapper before export to ensure it works
    print("\nTesting wrapper before export...")
    with torch.no_grad():
        test_out_acc, test_out_gyro, test_cov_acc, test_cov_gyro = wrapper(acc, gyro)
    print(f"  ✓ Test passed - output shapes:")
    print(f"    corrections: {test_out_acc.shape}, {test_out_gyro.shape}")
    print(f"    covariance:  {test_cov_acc.shape}, {test_cov_gyro.shape}")
    
    # Check if covariance is enabled
    has_cov = hasattr(net.conf, 'propcov') and net.conf.propcov
    if has_cov:
        print(f"  ✓ Covariance enabled (propcov=True)")
    else:
        print(f"  ℹ Covariance disabled - outputs will be zeros")
    
    # ============================================================================
    # EXPORT TO ONNX
    # ============================================================================
    print(f"\nExporting to ONNX (opset {opset})...")
    torch.onnx.export(
        wrapper,
        (acc, gyro),
        onnx_path,
        input_names=["acc", "gyro"],
        output_names=["corr_acc", "corr_gyro", "cov_acc", "cov_gyro"],
        opset_version=opset,
        do_constant_folding=True,
        training=torch.onnx.TrainingMode.EVAL,
        verbose=False
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
        
        # Prepare inputs in correct dtype
        acc_input = acc.numpy()
        gyro_input = gyro.numpy()
        
        # Run inference with dummy inputs
        onnx_outputs = sess.run(
            None,  # Return all outputs
            {"acc": acc_input, "gyro": gyro_input}
        )
        
        # Compare ONNX output with PyTorch output
        torch_outputs = [
            test_out_acc.numpy(), 
            test_out_gyro.numpy(),
            test_cov_acc.numpy(),
            test_cov_gyro.numpy()
        ]
        
        diff_acc = abs(onnx_outputs[0] - torch_outputs[0]).max()
        diff_gyro = abs(onnx_outputs[1] - torch_outputs[1]).max()
        diff_cov_acc = abs(onnx_outputs[2] - torch_outputs[2]).max()
        diff_cov_gyro = abs(onnx_outputs[3] - torch_outputs[3]).max()
        
        print(f"  Max difference (corr_acc):  {diff_acc:.2e}")
        print(f"  Max difference (corr_gyro): {diff_gyro:.2e}")
        print(f"  Max difference (cov_acc):   {diff_cov_acc:.2e}")
        print(f"  Max difference (cov_gyro):  {diff_cov_gyro:.2e}")
        
        # Check if differences are acceptable for fp32
        threshold = 1e-4
        max_diff = max(diff_acc, diff_gyro, diff_cov_acc, diff_cov_gyro)
        
        if max_diff < threshold:
            print("  ✓ ONNX export verified - all outputs match PyTorch!")
        else:
            print(f"  ⚠️  Max difference detected: {max_diff:.2e}")
            
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
    print(f"\nONNX Model:")
    print(f"  Precision: float32")
    print(f"  Inputs:  [1, {total_seq_len}, 3] ({num_imu_frames} real + {padding_frames} padding)")
    print(f"  Outputs: [1, {num_imu_frames}, 3] x 4")
    print(f"    - corr_acc:  IMU correction for accelerometer")
    print(f"    - corr_gyro: IMU correction for gyroscope")
    print(f"    - cov_acc:   Covariance for accelerometer")
    print(f"    - cov_gyro:  Covariance for gyroscope")
    print(f"\nFiles:")
    print(f"  ONNX:    {onnx_path}")
    print(f"  PyTorch: {torch_path}")
    print("="*70)


def main() -> None:
    """
    Export AirIMU model to ONNX format.
    
    You only need to specify the number of real IMU frames to process.
    Padding frames are added automatically by the model.
    """
    parser = argparse.ArgumentParser(
        description="Export AirIMU to ONNX with fixed input shape",
        formatter_class=argparse.RawDescriptionHelpFormatter
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
                        help="Output path for PyTorch weights")
    
    parser.add_argument("--opset", 
                        type=int, 
                        default=17, 
                        help="ONNX opset version")
    
    parser.add_argument("--num-imu-frames",
                        type=int,
                        default=1,
                        help="Number of real IMU frames to process (padding added automatically)")
    
    args = parser.parse_args()
    
    # Print configuration
    print("\n" + "="*70)
    print("AirIMU ONNX Export")
    print("="*70)
    print(f"Config:        {args.config}")
    print(f"Checkpoint:    {args.ckpt}")
    print(f"Output:        {args.onnx}")
    print(f"Opset:         {args.opset}")
    print(f"IMU frames:    {args.num_imu_frames}")
    print("="*70 + "\n")

    export(args.config, args.ckpt, args.onnx, args.torch, args.opset, args.num_imu_frames)


if __name__ == "__main__":
    main()