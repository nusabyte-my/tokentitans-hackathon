"""
GuardianBox - Temporal Video Action Dataset Generator
Extracts sliding-window video features and motion kinematics from the 6 real demo clips:
1. Walking.mp4 -> WALKING_AMBULATION (Safe)
2. Sujud ( Pray ).mp4 -> ISLAMIC_PRAYER_SUJUD (Safe / Cultural)
3. Acute Fall.mp4 -> ACUTE_SLIP_FALL (Critical Emergency)
4. Fall 1 Man.mp4 -> ACUTE_PRONE_FALL (Critical Emergency)
5. woman fall.mp4 -> LATERAL_IMPACT_FALL (Critical Emergency)
6. Chokin and Fall.mp4 -> CHOKING_AND_COLLAPSE (Critical Emergency)

Generates instruction-tuning datasets formatted for LLM/VLM Action Classification.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Tuple
import cv2
import numpy as np


@dataclass
class VideoActionMetadata:
    action_label: str
    is_emergency: bool
    triage_code: str
    cultural_context: Optional[str]
    standard_reasoning: str


VIDEO_ANNOTATIONS: Dict[str, VideoActionMetadata] = {
    "Walking.mp4": VideoActionMetadata(
        action_label="WALKING_AMBULATION",
        is_emergency=False,
        triage_code="NORMAL",
        cultural_context=None,
        standard_reasoning="Subject walking upright with vertical torso alignment and rhythmic bilateral gait.",
    ),
    "Sujud ( Pray ).mp4": VideoActionMetadata(
        action_label="ISLAMIC_PRAYER_SUJUD",
        is_emergency=False,
        triage_code="CULTURAL_CLEAR",
        cultural_context="Solat Prostration (Sujud on prayer mat)",
        standard_reasoning="Controlled kneeling and forehead prostration on prayer mat with symmetrical joint stabilization.",
    ),
    "Acute Fall.mp4": VideoActionMetadata(
        action_label="ACUTE_SLIP_FALL",
        is_emergency=True,
        triage_code="KOD MERAH (Red Zone)",
        cultural_context=None,
        standard_reasoning="Loss of balance resulting in acute slip and horizontal floor impact.",
    ),
    "Fall 1 Man.mp4": VideoActionMetadata(
        action_label="ACUTE_PRONE_FALL",
        is_emergency=True,
        triage_code="KOD MERAH (Red Zone)",
        cultural_context=None,
        standard_reasoning="High downward velocity collapse; subject prone on floor with flaccid limb splay.",
    ),
    "woman fall.mp4": VideoActionMetadata(
        action_label="LATERAL_IMPACT_FALL",
        is_emergency=True,
        triage_code="KOD MERAH (Red Zone)",
        cultural_context=None,
        standard_reasoning="Sideways lateral fall with severe hip impact hazard against floor surface.",
    ),
    "Chokin and Fall.mp4": VideoActionMetadata(
        action_label="CHOKING_AND_COLLAPSE",
        is_emergency=True,
        triage_code="KOD MERAH (Red Zone)",
        cultural_context=None,
        standard_reasoning="Senior clutching throat in respiratory distress followed by gradual catastrophic collapse.",
    ),
}


class VideoActionDatasetGenerator:
    def __init__(self, sample_videos_dir: Optional[str] = None):
        base_dir = os.path.dirname(os.path.dirname(__file__))
        self.sample_videos_dir = sample_videos_dir or os.path.join(base_dir, "sample_videos")
        self.output_dir = os.path.join(base_dir, "data")
        os.makedirs(self.output_dir, exist_ok=True)

    def extract_temporal_features_from_video(
        self,
        video_path: str,
        window_size: int = 15,
        stride: int = 8,
    ) -> List[Dict[str, Any]]:
        """Extracts rolling window kinematics and optical movement from video."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []

        mog2 = cv2.createBackgroundSubtractorMOG2(history=30, varThreshold=25, detectShadows=False)
        frame_features = []

        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            h, w = frame.shape[:2]
            frame_resized = cv2.resize(frame, (320, 240))

            fg = mog2.apply(frame_resized)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            largest_box = None
            max_area = 0
            for c in contours:
                area = cv2.contourArea(c)
                if area > 300 and area > max_area:
                    max_area = area
                    largest_box = cv2.boundingRect(c)

            if largest_box:
                bx, by, bw, bh = largest_box
                ar = float(bw) / max(float(bh), 1.0)
                y_center = (by + bh / 2.0) / 240.0
                torso_est = min(90.0, max(0.0, ar * 45.0 if ar > 1.2 else (1.0 - bh / max(bw, 1)) * 30.0))
            else:
                ar = 0.45
                y_center = 0.5
                torso_est = 5.0

            frame_features.append({
                "aspect_ratio": round(ar, 2),
                "torso_angle": round(torso_est, 1),
                "y_center": round(y_center, 2),
                "motion_area": int(max_area),
            })

        cap.release()

        # Build sliding windows
        windows = []
        for i in range(0, max(1, len(frame_features) - window_size + 1), stride):
            win = frame_features[i : i + window_size]
            if len(win) < 5:
                continue

            ar_vals = [f["aspect_ratio"] for f in win]
            torso_vals = [f["torso_angle"] for f in win]
            y_vals = [f["y_center"] for f in win]

            ar_mean = float(np.mean(ar_vals))
            ar_max = float(np.max(ar_vals))
            torso_mean = float(np.mean(torso_vals))
            torso_max = float(np.max(torso_vals))
            y_delta = float(y_vals[-1] - y_vals[0])  # positive means downward motion
            velocity_y = float(y_delta / max(len(win), 1))

            windows.append({
                "mean_ar": round(ar_mean, 2),
                "max_ar": round(ar_max, 2),
                "mean_torso_angle": round(torso_mean, 1),
                "max_torso_angle": round(torso_max, 1),
                "downward_descent": round(y_delta, 2),
                "vertical_velocity": round(velocity_y, 4),
                "temporal_length": len(win),
            })

        return windows

    def generate_instruction_dataset(self) -> Dict[str, Any]:
        """Processes all 6 demo videos and builds complete train/val dataset."""
        samples: List[Dict[str, Any]] = []

        for filename, meta in VIDEO_ANNOTATIONS.items():
            video_path = os.path.join(self.sample_videos_dir, filename)
            if not os.path.exists(video_path):
                continue

            windows = self.extract_temporal_features_from_video(video_path)
            for w_idx, win in enumerate(windows):
                input_desc = (
                    f"Temporal Video Sequence Window #{w_idx+1}:\n"
                    f"- Mean Aspect Ratio (w/h): {win['mean_ar']}, Peak AR: {win['max_ar']}\n"
                    f"- Mean Torso Angle: {win['mean_torso_angle']}°, Peak Torso: {win['max_torso_angle']}°\n"
                    f"- Downward Descent ΔY: {win['downward_descent']} (Velocity: {win['vertical_velocity']})\n"
                    f"- Temporal Sample Count: {win['temporal_length']} frames"
                )

                output_json = {
                    "action": meta.action_label,
                    "is_emergency": meta.is_emergency,
                    "confidence": 0.98 if meta.is_emergency else 0.96,
                    "triage_code": meta.triage_code,
                    "cultural_context": meta.cultural_context,
                    "reasoning": meta.standard_reasoning,
                }

                samples.append({
                    "instruction": "Classify the temporal video action sequence for sovereign geriatric home safety on NVIDIA DGX Spark.",
                    "input": input_desc,
                    "output": json.dumps(output_json),
                    "features": [
                        win["mean_ar"],
                        win["max_ar"],
                        win["mean_torso_angle"] / 90.0,
                        win["max_torso_angle"] / 90.0,
                        win["downward_descent"],
                        win["vertical_velocity"] * 10.0,
                    ],
                    "label_idx": list(VIDEO_ANNOTATIONS.keys()).index(filename),
                    "label_name": meta.action_label,
                    "is_emergency": meta.is_emergency,
                })

        # Save train/val split
        np.random.seed(42)
        shuffled = np.random.permutation(len(samples))
        val_count = max(4, int(len(samples) * 0.2))
        val_indices = set(shuffled[:val_count])

        train_set = [samples[i] for i in range(len(samples)) if i not in val_indices]
        val_set = [samples[i] for i in range(len(samples)) if i in val_indices]

        output_path = os.path.join(self.output_dir, "video_action_dataset.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({
                "metadata": {
                    "total_samples": len(samples),
                    "train_count": len(train_set),
                    "val_count": len(val_set),
                    "classes": [m.action_label for m in VIDEO_ANNOTATIONS.values()],
                    "target_platform": "Gigabyte AI TOP ATOM (128GB LPDDR5x)",
                },
                "train": train_set,
                "val": val_set,
            }, f, indent=2)

        return {
            "total_samples": len(samples),
            "train_samples": len(train_set),
            "val_samples": len(val_set),
            "dataset_file": output_path,
        }


if __name__ == "__main__":
    generator = VideoActionDatasetGenerator()
    res = generator.generate_instruction_dataset()
    print("Generated Video Action Dataset:")
    print(f"  • Total Samples: {res['total_samples']}")
    print(f"  • Train Samples: {res['train_samples']}")
    print(f"  • Val Samples: {res['val_samples']}")
    print(f"  • Saved: {res['dataset_file']}")
