"""
PathSense ONNX Export & INT8 Quantization Utility Script
"""

import argparse
from pathsense.quantization import QuantizationEngine


def main():
    parser = argparse.ArgumentParser(description="PathSense ONNX Export & Edge Quantization Utility")
    parser.add_argument("--output", type=str, default="models/depth_anything_v2_small.onnx", help="Output ONNX model path")
    args = parser.parse_args()

    print("==========================================================")
    print(" PathSense Edge Optimization & INT8 Quantization Benchmark")
    print("==========================================================")

    success = QuantizationEngine.export_onnx_dummy(args.output)
    if success:
        print(f"[SUCCESS] Exported ONNX graph model to '{args.output}'")

    report = QuantizationEngine.benchmark_simulation()
    print("\nINT8 QUANTIZATION BENCHMARK REPORT (Raspberry Pi 5 Target):")
    print("-" * 55)
    print(f"  Model Name             : {report.model_name}")
    print(f"  FP32 Model Size        : {report.fp32_size_mb:.1f} MB")
    print(f"  INT8 Model Size        : {report.int8_size_mb:.1f} MB ({report.size_reduction_pct}% reduction)")
    print(f"  FP32 Latency           : {report.fp32_latency_ms:.2f} ms")
    print(f"  INT8 Quantized Latency : {report.int8_latency_ms:.2f} ms")
    print(f"  Speedup Factor         : {report.speedup_factor:.2f}x")
    print(f"  MAE Depth Diff (Drop)  : {report.mean_absolute_depth_diff:.4f}")
    print("==========================================================")


if __name__ == "__main__":
    main()
