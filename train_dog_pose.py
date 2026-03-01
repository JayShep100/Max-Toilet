#!/usr/bin/env python3
"""
train_dog_pose.py — Fine-tune a YOLO pose model on the Ultralytics Dog-Pose Dataset.
=======================================================================================
The trained model provides 24 anatomically correct dog keypoints that are used by
``smart_reviewer.py`` and ``stage3_trainer.py`` for wee/poo/neither classification.

Dog-Pose 24-keypoint skeleton:
  0: front_left_paw      1: front_left_knee     2: front_left_elbow
  3: rear_left_paw       4: rear_left_knee       5: rear_left_elbow
  6: front_right_paw     7: front_right_knee     8: front_right_elbow
  9: rear_right_paw     10: rear_right_knee     11: rear_right_elbow
 12: tail_start         13: tail_end
 14: left_ear_base      15: right_ear_base
 16: nose               17: chin
 18: left_ear_tip       19: right_ear_tip
 20: left_eye           21: right_eye
 22: withers            23: throat

Usage:
    # Train with defaults (yolo11n-pose.pt base, 100 epochs)
    python3 train_dog_pose.py

    # Custom settings
    python3 train_dog_pose.py --base-model yolo11s-pose.pt --epochs 200 --imgsz 640

    # Specify output path
    python3 train_dog_pose.py --output my_dog_pose.pt

Requirements:
    pip install ultralytics
"""

import argparse
import sys
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    sys.exit("pip install ultralytics")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune a YOLO pose model on the Ultralytics Dog-Pose Dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--base-model", default="yolo11n-pose.pt",
        help="Base YOLO pose model to fine-tune (default: yolo11n-pose.pt).",
    )
    parser.add_argument(
        "--epochs", type=int, default=100,
        help="Number of training epochs (default: 100).",
    )
    parser.add_argument(
        "--imgsz", type=int, default=640,
        help="Input image size in pixels (default: 640).",
    )
    parser.add_argument(
        "--output", default="dog_pose_model.pt",
        help="Output path for the trained model weights (default: dog_pose_model.pt).",
    )
    args = parser.parse_args()

    output_path = Path(args.output)

    print(f"Loading base model: {args.base_model}")
    model = YOLO(args.base_model)

    print(
        f"Training on Dog-Pose dataset for {args.epochs} epoch(s) "
        f"at image size {args.imgsz}…"
    )
    print("(The dataset will be downloaded automatically on first run.)")

    # The 'dog-pose' dataset YAML is bundled with ultralytics.
    # It downloads ~290 MB of images on first run.
    results = model.train(
        data="dog-pose.yaml",
        epochs=args.epochs,
        imgsz=args.imgsz,
        project="dog_pose_training",
        name="train",
        exist_ok=True,
    )

    # Export / copy the best weights to the requested output path.
    best_weights = Path(results.save_dir) / "weights" / "best.pt"
    if best_weights.exists():
        import shutil
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_weights, output_path)
        print(f"\n✅ Dog-pose model saved to: {output_path}")
    else:
        # Fall back to the last checkpoint if best.pt is missing.
        last_weights = Path(results.save_dir) / "weights" / "last.pt"
        if last_weights.exists():
            import shutil
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(last_weights, output_path)
            print(f"\n✅ Dog-pose model (last checkpoint) saved to: {output_path}")
        else:
            print(
                "\n⚠  Could not locate best.pt or last.pt in the training output. "
                f"Check the 'dog_pose_training/train/weights/' directory manually."
            )
            sys.exit(1)

    print("\nNext steps:")
    print(f"  # Review clips with the new dog-pose model:")
    print(f"  python3 smart_reviewer.py --pad-clips pad_clips_combined.csv \\")
    print(f"      --dest-root /path/to/labels --pose-model {output_path}")
    print()
    print(f"  # Train the toilet classifier using dog-pose features:")
    print(f"  python3 stage3_trainer.py --labelled /path/to/labels \\")
    print(f"      --labels labels.csv --pose-model {output_path}")


if __name__ == "__main__":
    main()
