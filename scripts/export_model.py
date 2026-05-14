#!/usr/bin/env python3
"""
Script 8: Export trained model to various formats
"""

import sys
import os
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np

from src.models.policy_net import PolicyNetwork
from src.utils.logger import setup_logger


def export_to_torchscript(model, input_shape, output_path):
    """Export to TorchScript"""
    print("   Exporting to TorchScript...")
    dummy_input = torch.randn(1, *input_shape)
    traced_model = torch.jit.trace(model, dummy_input)
    traced_model.save(output_path)
    return output_path


def export_to_onnx(model, input_shape, output_path):
    """Export to ONNX format"""
    print("   Exporting to ONNX...")
    dummy_input = torch.randn(1, *input_shape)
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
        opset_version=14
    )
    return output_path


def main():
    parser = argparse.ArgumentParser(description='Export trained model')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--output-dir', type=str, default='experiments/export',
                        help='Directory to save exported models')
    parser.add_argument('--format', type=str, default='all',
                        choices=['pt', 'torchscript', 'onnx', 'all'],
                        help='Export format')
    parser.add_argument('--input-dim', type=int, default=115,
                        help='Input feature dimension')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Model Export")
    print("=" * 60)
    
    # Setup device
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"Input dimension: {args.input_dim}")
    
    # Load model
    print(f"\n📂 Loading model from: {args.model}")
    if not Path(args.model).exists():
        print(f"❌ Model not found: {args.model}")
        sys.exit(1)
    
    model = PolicyNetwork()
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()
    model.to(device)
    print("✅ Model loaded")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Export formats
    input_shape = (args.input_dim,)
    base_name = Path(args.model).stem
    
    if args.format in ['pt', 'all']:
        # Original PyTorch format
        pt_path = output_dir / f"{base_name}.pt"
        torch.save(model.state_dict(), pt_path)
        print(f"✅ PyTorch model saved: {pt_path}")
    
    if args.format in ['torchscript', 'all']:
        # TorchScript format
        ts_path = output_dir / f"{base_name}.pt"
        export_to_torchscript(model, input_shape, ts_path)
        print(f"✅ TorchScript model saved: {ts_path}")
    
    if args.format in ['onnx', 'all']:
        # ONNX format
        onnx_path = output_dir / f"{base_name}.onnx"
        export_to_onnx(model, input_shape, onnx_path)
        print(f"✅ ONNX model saved: {onnx_path}")
    
    # Model info
    print("\n" + "=" * 60)
    print("Model Information")
    print("=" * 60)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Input shape: (batch_size, {args.input_dim})")
    print(f"Output shape: (batch_size, 6)")
    
    print(f"\n✅ Export complete! Files saved to: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()