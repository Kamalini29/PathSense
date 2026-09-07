"""
PathSense CLI Demo Runner
Executes PathSense pipeline on live camera feed, video file, or synthetic frames.
"""

import argparse
import time
import cv2
import numpy as np

from pathsense import PathSensePipeline, SystemConfig


def main():
    parser = argparse.ArgumentParser(description="PathSense Real-Time Navigation Demo Runner")
    parser.add_argument("--source", type=str, default="synthetic", help="Video source: '0' for webcam, file path, or 'synthetic'")
    parser.add_argument("--synthetic", action="store_true", help="Force synthetic monocular depth & detector fallback")
    parser.add_argument("--no-display", action="store_true", help="Headless mode (don't pop up OpenCV window)")
    parser.add_argument("--max-frames", type=int, default=100, help="Max frames to process in demo mode")
    args = parser.parse_args()

    print("==========================================================")
    print(" PathSense Monocular Depth & Navigation Assistant Demo")
    print("==========================================================")

    config = SystemConfig()
    pipeline = PathSensePipeline(config=config, force_synthetic=args.synthetic or (args.source == "synthetic"))

    if args.source == "synthetic":
        print("[INFO] Running in Synthetic Multi-Scenario Demo Mode...")
        for frame_idx in range(1, args.max_frames + 1):
            # Create synthetic test frame with shifting obstacles
            frame = np.full((480, 640, 3), (40, 30, 20), dtype=np.uint8)
            
            # Scenario 1: Obstacle on Right (Frames 1-30)
            if frame_idx <= 30:
                cv2.rectangle(frame, (420, 200), (600, 460), (0, 0, 220), -1) # Red hazard block right
            # Scenario 2: Obstacle Center (Frames 31-60)
            elif frame_idx <= 60:
                cv2.rectangle(frame, (220, 180), (420, 460), (0, 0, 220), -1) # Red hazard block center
            # Scenario 3: Obstacle Left (Frames 61-90)
            else:
                cv2.rectangle(frame, (40, 200), (220, 460), (0, 0, 220), -1) # Red hazard block left

            output = pipeline.process_frame(frame, trigger_audio=False)
            
            print(f"Frame [{frame_idx:03d}/{args.max_frames:03d}] -> Action: {output.decision.action.value:<15} | Spoken: '{output.decision.spoken_phrase}' | FPS: {output.latency.fps:.1f}")

            if not args.no_display:
                cv2.imshow("PathSense Live Feed", output.annotated_frame)
                cv2.imshow("Depth Anything V2 Heatmap", output.depth_heatmap)
                if cv2.waitKey(30) & 0xFF == 27:
                    break

    else:
        video_src = int(args.source) if args.source.isdigit() else args.source
        cap = cv2.VideoCapture(video_src)
        if not cap.isOpened():
            print(f"[ERROR] Could not open video source '{args.source}'")
            return

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1

            output = pipeline.process_frame(frame, trigger_audio=True)
            print(f"Frame [{frame_count:04d}] -> Action: {output.decision.action.value:<15} | FPS: {output.latency.fps:.1f}")

            if not args.no_display:
                cv2.imshow("PathSense Live Feed", output.annotated_frame)
                cv2.imshow("Depth Anything V2 Heatmap", output.depth_heatmap)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

        cap.release()

    if not args.no_display:
        cv2.destroyAllWindows()
    print("[INFO] PathSense Demo finished successfully.")


if __name__ == "__main__":
    main()
