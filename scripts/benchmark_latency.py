"""
PathSense Latency Profiling Benchmark
Runs pipeline over N test iterations and computes mean latency per stage.
"""

import argparse
import numpy as np

from pathsense import PathSensePipeline, SystemConfig


def main():
    parser = argparse.ArgumentParser(description="PathSense Latency Profiler Benchmark")
    parser.add_argument("--iterations", type=int, default=50, help="Number of benchmark iterations")
    parser.add_argument("--synthetic", action="store_true", default=True, help="Use synthetic fallback engine")
    args = parser.parse_args()

    print("==========================================================")
    print(f" PathSense Pipeline Latency Benchmark ({args.iterations} iterations)")
    print("==========================================================")

    pipeline = PathSensePipeline(force_synthetic=args.synthetic)
    dummy_frame = np.full((480, 640, 3), 128, dtype=np.uint8)

    # Warmup runs
    for _ in range(5):
        pipeline.process_frame(dummy_frame, trigger_audio=False)

    stage_accum = {
        "capture": [],
        "depth": [],
        "detect": [],
        "fusion": [],
        "decision": [],
        "audio": [],
        "total": [],
    }

    for idx in range(args.iterations):
        out = pipeline.process_frame(dummy_frame, trigger_audio=False)
        lat = out.latency
        stage_accum["capture"].append(lat.capture_ms)
        stage_accum["depth"].append(lat.depth_ms)
        stage_accum["detect"].append(lat.detect_ms)
        stage_accum["fusion"].append(lat.fusion_ms)
        stage_accum["decision"].append(lat.decision_ms)
        stage_accum["audio"].append(lat.audio_ms)
        stage_accum["total"].append(lat.total_ms)

    print("\nSTAGE LATENCY BREAKDOWN (ms):")
    print("-" * 50)
    for stage, vals in stage_accum.items():
        if stage == "total":
            continue
        mean_val = np.mean(vals)
        std_val = np.std(vals)
        print(f"  Stage [{stage.upper():<10}]: {mean_val:6.2f} ms ± {std_val:.2f} ms")

    total_mean = np.mean(stage_accum["total"])
    fps_mean = 1000.0 / total_mean if total_mean > 0 else 0
    print("-" * 50)
    print(f"  TOTAL LATENCY    : {total_mean:6.2f} ms")
    print(f"  THROUGHPUT (FPS) : {fps_mean:6.1f} FPS")
    print("==========================================================")


if __name__ == "__main__":
    main()
