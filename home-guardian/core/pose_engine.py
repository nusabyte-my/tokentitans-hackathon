"""
GuardianBox - Pose Engine & Fall Geometry Detector
High-frequency (30 FPS) edge computer vision pipeline.
Extracts skeletal landmarks, calculates torso angle, hip-to-ground ratio,
and manages temporal debounce buffers before escalating to Nemotron Omni.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np


@dataclass
class Keypoint:
    x: float
    y: float
    score: float = 1.0


@dataclass
class PoseDetectionResult:
    is_detected: bool
    is_on_floor: bool
    state: str  # "NORMAL", "SUSPECT", "FALL", "PRAYER_LIKELY", "RESTING"
    confidence: float
    aspect_ratio: float
    torso_angle: float
    hip_height_ratio: float
    horizontal_duration_sec: float
    keypoints: Dict[str, Tuple[int, int]] = field(default_factory=dict)
    bounding_box: Optional[Tuple[int, int, int, int]] = None  # (x, y, w, h)


class PoseEngine:
    def __init__(
        self,
        floor_seconds_threshold: float = 3.0,
        aspect_ratio_threshold: float = 1.25,
        torso_angle_threshold: float = 40.0,
    ):
        """
        floor_seconds_threshold: Seconds a person must stay horizontal to trigger SUSPECT
        aspect_ratio_threshold: Width / Height of bounding box (>1.25 indicates horizontal posture)
        torso_angle_threshold: Degrees from vertical (>40 indicates leaning/floor posture)
        """
        self.floor_seconds_threshold = floor_seconds_threshold
        self.aspect_ratio_threshold = aspect_ratio_threshold
        self.torso_angle_threshold = torso_angle_threshold

        self.horizontal_start_time: Optional[float] = None
        self.current_state: str = "NORMAL"
        self.last_detection_time: float = time.time()
        self.frame_count: int = 0

    def analyze_geometry(
        self,
        bbox: Tuple[int, int, int, int],
        frame_shape: Tuple[int, int],
        custom_keypoints: Optional[Dict[str, Tuple[int, int]]] = None,
    ) -> PoseDetectionResult:
        """
        Analyzes bounding box & landmarks to determine posture geometry.
        bbox: (x, y, w, h)
        frame_shape: (height, width)
        """
        x, y, w, h = bbox
        frame_h, frame_w = frame_shape[:2]

        aspect_ratio = float(w) / max(float(h), 1.0)
        hip_height_ratio = float(y + h) / float(frame_h)  # closer to 1.0 means bottom of frame

        # Calculate approximate torso angle from vertical axis (0 deg = upright)
        torso_angle = 0.0
        if custom_keypoints and "nose" in custom_keypoints and "hip" in custom_keypoints:
            nx, ny = custom_keypoints["nose"]
            hx, hy = custom_keypoints["hip"]
            dx = abs(nx - hx)
            dy = max(hy - ny, 1)  # hy > ny when upright
            torso_angle = abs(math.degrees(math.atan2(dx, dy)))
        else:
            # Fallback estimation based on aspect ratio
            if aspect_ratio > 1.2:
                torso_angle = min(90.0, aspect_ratio * 45.0)
            else:
                torso_angle = max(0.0, (1.0 - (h / max(w, 1))) * 30.0)

        # Condition for being horizontal on floor
        is_horizontal = (aspect_ratio >= self.aspect_ratio_threshold) or (torso_angle >= self.torso_angle_threshold)
        
        # Debounce logic
        now = time.time()
        horizontal_duration = 0.0

        if is_horizontal:
            if self.horizontal_start_time is None:
                self.horizontal_start_time = now
            horizontal_duration = now - self.horizontal_start_time
            if horizontal_duration >= self.floor_seconds_threshold:
                self.current_state = "SUSPECT"
            else:
                self.current_state = "EVALUATING"
        else:
            # Stood up or recovery
            self.horizontal_start_time = None
            self.current_state = "NORMAL"

        confidence = min(0.99, max(0.65, aspect_ratio / 2.0))

        return PoseDetectionResult(
            is_detected=True,
            is_on_floor=is_horizontal,
            state=self.current_state,
            confidence=confidence,
            aspect_ratio=round(aspect_ratio, 2),
            torso_angle=round(torso_angle, 1),
            hip_height_ratio=round(hip_height_ratio, 2),
            horizontal_duration_sec=round(horizontal_duration, 1),
            keypoints=custom_keypoints or {},
            bounding_box=bbox,
        )

    def draw_skeleton(
        self,
        frame: np.ndarray,
        result: PoseDetectionResult,
        color_scheme: str = "auto",
    ) -> np.ndarray:
        """
        Draws tactical hairline skeleton and bounding box on frame.
        """
        annotated = frame.copy()
        if not result.is_detected or not result.bounding_box:
            return annotated

        x, y, w, h = result.bounding_box

        # Tactical Hairline Colors (BGR)
        if result.state == "NORMAL":
            color = (80, 200, 120)  # Emerald green
            status_text = "NORMAL (UPRIGHT)"
        elif result.state == "EVALUATING":
            color = (30, 180, 240)  # Amber gold
            status_text = f"ANALYZING ({result.horizontal_duration_sec:.1f}s)"
        elif result.state == "SUSPECT":
            color = (60, 60, 240)  # Vibrant crimson
            status_text = f"SUSPECT: FLOOR DETECTED ({result.horizontal_duration_sec:.1f}s)"
        else:
            color = (180, 180, 180)
            status_text = result.state

        # Hairline bounding box (1px thin)
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 1, cv2.LINE_AA)

        # Tactical Corner Accents
        c_len = min(15, w // 4, h // 4)
        cv2.line(annotated, (x, y), (x + c_len, y), color, 2, cv2.LINE_AA)
        cv2.line(annotated, (x, y), (x, y + c_len), color, 2, cv2.LINE_AA)
        cv2.line(annotated, (x + w, y), (x + w - c_len, y), color, 2, cv2.LINE_AA)
        cv2.line(annotated, (x + w, y), (x + w, y + c_len), color, 2, cv2.LINE_AA)
        cv2.line(annotated, (x, y + h), (x + c_len, y + h), color, 2, cv2.LINE_AA)
        cv2.line(annotated, (x, y + h), (x, y + h - c_len), color, 2, cv2.LINE_AA)
        cv2.line(annotated, (x + w, y + h), (x + w - c_len, y + h), color, 2, cv2.LINE_AA)
        cv2.line(annotated, (x + w, y + h), (x + w, y + h - c_len), color, 2, cv2.LINE_AA)

        # Hairline Skeleton Joints
        kpts = result.keypoints
        for k_name, (kx, ky) in kpts.items():
            cv2.circle(annotated, (kx, ky), 3, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(annotated, (kx, ky), 4, color, 1, cv2.LINE_AA)

        # Connect limbs if present
        connections = [
            ("nose", "neck"),
            ("neck", "hip"),
            ("neck", "left_shoulder"),
            ("neck", "right_shoulder"),
            ("left_shoulder", "left_elbow"),
            ("right_shoulder", "right_elbow"),
            ("hip", "left_knee"),
            ("hip", "right_knee"),
            ("left_knee", "left_foot"),
            ("right_knee", "right_foot"),
        ]
        for p1, p2 in connections:
            if p1 in kpts and p2 in kpts:
                cv2.line(annotated, kpts[p1], kpts[p2], color, 1, cv2.LINE_AA)

        # Status HUD Pill on top of BBox
        label = f"{status_text} | AR: {result.aspect_ratio} | {result.torso_angle}deg"
        cv2.putText(
            annotated,
            label,
            (x, max(20, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )

        return annotated

    def reset(self):
        """Reset state machine"""
        self.horizontal_start_time = None
        self.current_state = "NORMAL"
