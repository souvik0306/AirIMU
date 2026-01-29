#!/usr/bin/env python3
"""
Run offline inference using an exported ONNX AirIMU model.

This script mimics inference.py but uses ONNX Runtime instead of PyTorch for inference.
The dataset loading, padding (via collate_fn), and output saving are identical to inference.py.

Key differences from inference.py:
- Loads ONNX model instead of PyTorch checkpoint
- Uses ONNX Runtime for forward pass
- Everything else (dataset, collate, padding) remains the same
"""
import os
import torch
import torch.utils.data as Data
import argparse
import pickle
import numpy as np
import tqdm
import onnxruntime as ort
from pyhocon import ConfigFactory

from utils import move_to, save_state
from datasets import collate_fcs, SeqeuncesDataset
from model import net_dict


def inference(onnx_session, loader, confs, interval=9, torch_net=None, verbose=False):
    """
    ONNX-based correction inference.
    
    This function replaces network.inference() with ONNX Runtime inference.
    The data flow remains identical to the PyTorch version.
    
    Args:
        onnx_session: ONNX Runtime InferenceSession
        loader: DataLoader with collate_fn that adds padding
        confs: Configuration object
        interval: Model interval (padding frames, typically 9)
    
    Returns:
        evaluate_states: Dictionary with correction results
    """
    # Detect ONNX model input dtype and shape
    input_dtype_str = onnx_session.get_inputs()[0].type
    is_fp16 = 'float16' in input_dtype_str
    
    # Get expected input sequence length from ONNX model
    onnx_input_shape = onnx_session.get_inputs()[0].shape
    expected_seq_len = onnx_input_shape[1]  # e.g., 1009 for 1000 real + 9 padding
    
    if is_fp16:
        print("⚠️  ONNX model uses float16 precision")
        np_dtype = np.float16
    else:
        print("✓ ONNX model uses float32 precision")
        np_dtype = np.float32
    
    print(f"✓ ONNX model expects fixed input shape: {onnx_input_shape}")
    
    evaluate_states = {}

    with torch.no_grad():
        inte_state = None
        first_batch = True
        compare_done = False
        for data, _, _ in tqdm.tqdm(loader):
            # Print padding info for first batch only
            if first_batch:
                print(f"\n=== Padding Check (First Batch) ===")
                print(f"Input acc shape after collate: {data['acc'].shape}")
                print(f"Input gyro shape after collate: {data['gyro'].shape}")
                if data['acc'].shape[1] > 1:
                    print(f"✓ PADDING DETECTED: Input has {data['acc'].shape[1]} samples")
                print(f"===================================\n")
                first_batch = False
            
            # Get actual sequence length from batch
            actual_seq_len = data['acc'].shape[1]
            actual_real_frames = actual_seq_len - interval
            
            # Pad batch if it's shorter than expected (handles last batch with fewer frames)
            if actual_seq_len < expected_seq_len:
                pad_amount = expected_seq_len - actual_seq_len
                # Pad with zeros at the end
                acc_padded = torch.cat([data['acc'], torch.zeros(1, pad_amount, 3)], dim=1)
                gyro_padded = torch.cat([data['gyro'], torch.zeros(1, pad_amount, 3)], dim=1)
            else:
                acc_padded = data['acc']
                gyro_padded = data['gyro']
            
            # Convert to numpy with correct dtype for ONNX model
            acc_np = acc_padded.cpu().numpy().astype(np_dtype)
            gyro_np = gyro_padded.cpu().numpy().astype(np_dtype)
            
            # Run ONNX inference
            # Input: [batch, expected_seq_len, 3] -> Output: 4x [batch, expected_seq_len-interval, 3]
            corr_acc_full, corr_gyro_full, cov_acc_full, cov_gyro_full = onnx_session.run(
                None,  # Return all outputs
                {"acc": acc_np, "gyro": gyro_np}
            )
            
            # Trim output to actual number of real frames (remove extra padding)
            corr_acc = corr_acc_full[:, :actual_real_frames, :]
            corr_gyro = corr_gyro_full[:, :actual_real_frames, :]
            cov_acc = cov_acc_full[:, :actual_real_frames, :]
            cov_gyro = cov_gyro_full[:, :actual_real_frames, :]

            # Optional debug: compare ONNX output with PyTorch model on first batch
            if (torch_net is not None) and (not compare_done):
                try:
                    # Prepare PyTorch inputs (match model precision)
                    if is_fp16:
                        data_torch = {
                            'acc': data['acc'].half(),
                            'gyro': data['gyro'].half()
                        }
                    else:
                        data_torch = {
                            'acc': data['acc'].float(),
                            'gyro': data['gyro'].float()
                        }
                    # Run PyTorch inference (model.inference expects dict)
                    with torch.no_grad():
                        pt_out = torch_net.inference(data_torch)

                    pt_corr_acc = pt_out['correction_acc'].cpu().numpy()
                    pt_corr_gyro = pt_out['correction_gyro'].cpu().numpy()
                    
                    # Get covariance from cov_state (may be None)
                    cov_state = pt_out.get('cov_state', {})
                    pt_cov_acc = cov_state.get('acc_cov')
                    pt_cov_gyro = cov_state.get('gyro_cov')
                    if pt_cov_acc is not None:
                        pt_cov_acc = pt_cov_acc.cpu().numpy()
                    if pt_cov_gyro is not None:
                        pt_cov_gyro = pt_cov_gyro.cpu().numpy()

                    onnx_acc = corr_acc
                    onnx_gyro = corr_gyro

                    diff_acc = np.max(np.abs(onnx_acc - pt_corr_acc))
                    diff_gyro = np.max(np.abs(onnx_gyro - pt_corr_gyro))
                    print(f"\n[DEBUG] First-batch ONNX vs PyTorch max abs diff:")
                    print(f"  Corrections - acc: {diff_acc:.3e}, gyro: {diff_gyro:.3e}")
                    
                    # Compare covariance if available
                    if pt_cov_acc is not None and pt_cov_gyro is not None:
                        diff_cov_acc = np.max(np.abs(cov_acc - pt_cov_acc))
                        diff_cov_gyro = np.max(np.abs(cov_gyro - pt_cov_gyro))
                        print(f"  Covariance  - acc: {diff_cov_acc:.3e}, gyro: {diff_cov_gyro:.3e}")
                    else:
                        print(f"  Covariance  - not enabled in model (outputs are zeros)")
                    
                    if verbose:
                        print(f"  ONNX shapes: acc={onnx_acc.shape}, gyro={onnx_gyro.shape}")
                        print(f"  PyTorch shapes: acc={pt_corr_acc.shape}, gyro={pt_corr_gyro.shape}")
                except Exception as e:
                    print(f"[DEBUG] PyTorch comparison failed: {e}")
                compare_done = True
            
            # Convert back to torch tensors (always use float64 for downstream processing)
            inte_state = {
                'correction_acc': torch.from_numpy(corr_acc.astype(np.float64)),
                'correction_gyro': torch.from_numpy(corr_gyro.astype(np.float64)),
                'acc_cov': torch.from_numpy(cov_acc.astype(np.float64)),
                'gyro_cov': torch.from_numpy(cov_gyro.astype(np.float64))
            }
            
            # Save state (accumulate across batches)
            save_state(evaluate_states, inte_state)

        # Concatenate all batches
        for k, v in evaluate_states.items():
            evaluate_states[k] = torch.cat(v, dim=-2)

    return evaluate_states


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/exp/EuRoC/codenet.conf', help='config file path')
    parser.add_argument('--onnx', type=str, required=True, help='path to ONNX model file')
    parser.add_argument('--ckpt', type=str, default=None, help='(optional) PyTorch checkpoint to compare outputs (debug)')
    parser.add_argument("--device", type=str, default="cpu", help="cpu only for ONNX")
    parser.add_argument('--batch_size', type=int, default=1, help='batch size.')
    parser.add_argument('--window_size', type=int, default=1000, help='window size for sliding window')
    parser.add_argument('--step_size', type=int, default=1000, help='step size for sliding window')
    parser.add_argument('--train', default=False, action="store_true", help='if True, evaluate the training set')
    parser.add_argument('--gtinit', default=True, action="store_false", help='if set False, use integrated pose')
    parser.add_argument('--whole', default=False, action="store_true", help='process entire sequence at once')
    parser.add_argument('--verbose', action='store_true', help='print debug info and compare ONNX vs PyTorch if --ckpt provided')

    args = parser.parse_args()
    print(args)
    
    # Load configuration
    conf = ConfigFactory.parse_file(args.config)
    conf.train.device = "cpu"  # ONNX runs on CPU
    conf_name = os.path.split(args.config)[-1].split(".")[0]
    conf['general']['exp_dir'] = os.path.join(conf.general.exp_dir, conf_name)
    conf.train['sampling'] = False
    conf["gtinit"] = args.gtinit
    conf['device'] = "cpu"

    # Load ONNX model
    print(f"\nLoading ONNX model from: {args.onnx}")
    if not os.path.exists(args.onnx):
        raise Exception(f"ONNX model not found: {args.onnx}")
    
    onnx_session = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    print("✓ ONNX model loaded successfully")
    
    # Print ONNX model info
    print("\nONNX Model I/O:")
    for inp in onnx_session.get_inputs():
        print(f"  Input: {inp.name}, shape: {inp.shape}, dtype: {inp.type}")
    for out in onnx_session.get_outputs():
        print(f"  Output: {out.name}, shape: {out.shape}, dtype: {out.type}")
    
    # Get interval from network config (for reference)
    network = net_dict[conf.train.network](conf.train)
    interval = getattr(network, "interval", 9)
    print(f"\nModel interval (padding frames): {interval}")

    # Optional: load PyTorch checkpoint for debug comparison with ONNX
    torch_net = None
    if args.ckpt is not None:
        ckpt_path = args.ckpt
        if not os.path.exists(ckpt_path):
            print(f"Warning: checkpoint not found: {ckpt_path} - skipping PyTorch comparison")
        else:
            print(f"Loading PyTorch checkpoint for comparison: {ckpt_path}")
            torch_net = net_dict[conf.train.network](conf.train)
            state = torch.load(ckpt_path, map_location='cpu')
            torch_net.load_state_dict(state['model_state_dict'])
            torch_net.eval()
            # Force float32 to match ONNX export default
            try:
                torch_net = torch_net.float()
            except Exception:
                pass
            print("✓ PyTorch model loaded (for ONNX comparison)")
    
    # Setup output directory
    save_folder = os.path.join(conf.general.exp_dir, "evaluate")
    os.makedirs(save_folder, exist_ok=True)

    # Get collate function (handles padding)
    if 'collate' in conf.dataset.keys():
        collate_fn = collate_fcs[conf.dataset.collate]
        print(f"Using collate function: {conf.dataset.collate}")
    else:
        collate_fn = collate_fcs['base']
        print("Using base collate function")
    
    print(conf.dataset)
    dataset_conf = conf.dataset.inference

    # Run inference on all sequences
    print("\n" + "="*80)
    print("Starting ONNX Inference")
    print("="*80)
    
    net_out_result = {}
    dataset_conf.data_list[0]["window_size"] = args.window_size
    dataset_conf.data_list[0]["step_size"] = args.step_size
    
    for data_conf in dataset_conf.data_list:
        for path in data_conf.data_drive:
            if args.whole:
                dataset_conf["mode"] = "inference"
            else:
                dataset_conf["mode"] = "infevaluate"
            dataset_conf["exp_dir"] = conf.general.exp_dir
            
            print("\n" + "="*80)
            print(f"Processing sequence: {path}")
            print("="*80)
            print(str(dataset_conf))
            
            # Create dataset and dataloader
            eval_dataset = SeqeuncesDataset(
                data_set_config=dataset_conf, 
                data_path=path, 
                data_root=data_conf["data_root"]
            )
            eval_loader = Data.DataLoader(
                dataset=eval_dataset, 
                batch_size=args.batch_size, 
                shuffle=False, 
                collate_fn=collate_fn, 
                drop_last=False
            )

            # Print dataset info
            print(f"\n--- Sequence: {path} ---")
            total_samples = eval_dataset.acc[0].shape[0]
            window_size = dataset_conf.data_list[0]['window_size']
            step_size = dataset_conf.data_list[0]['step_size']
            overlap = window_size - step_size
            print(f"Total IMU samples: {total_samples}")
            print(f"Window size: {window_size}, Step size: {step_size}, Overlap: {overlap}")
            print(f"Number of windows: {len(eval_dataset)}")

            # Run ONNX inference
            inference_state = inference(
                onnx_session=onnx_session, 
                loader=eval_loader, 
                confs=conf.train,
                interval=interval,
                torch_net=torch_net,
                verbose=args.verbose
            )

            # Print output info
            print("\n" + "="*80)
            print("Inference Results")
            print("="*80)
            for k, v in inference_state.items():
                if hasattr(v, 'shape'):
                    print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
            
            # Apply corrections to original IMU data
            inference_state['corrected_acc'] = eval_dataset.acc[0] + inference_state['correction_acc'].squeeze(0).cpu()
            inference_state['corrected_gyro'] = eval_dataset.gyro[0] + inference_state['correction_gyro'].squeeze(0).cpu()
            inference_state['rot'] = eval_dataset.gt_ori[0]
            inference_state['dt'] = eval_dataset.dt[0]
            
            net_out_result[path] = inference_state
            print(f"✓ Completed: {inference_state['corrected_acc'].shape[0]} corrected samples")

    # Save results in pickle format (same as inference.py)
    net_result_path = os.path.join(conf.general.exp_dir, 'net_output_onnx.pickle')
    print("\n" + "="*80)
    print(f"Saving results to: {net_result_path}")
    with open(net_result_path, 'wb') as handle:
        pickle.dump(net_out_result, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("✓ Saved ONNX inference results")
    print("="*80)
    
    print("\n✓ ONNX inference complete!")
    print(f"\nOutput pickle: {net_result_path}")
    print(f"\nCompare with PyTorch inference:")
    print(f"  python inference.py --config {args.config}")
    print(f"\nEvaluate results:")
    print(f"  python evaluation/evaluate_state.py --pytorch {conf.general.exp_dir}/net_output.pickle --onnx {net_result_path}")
