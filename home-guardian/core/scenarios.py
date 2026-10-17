"""
GuardianBox - Deterministic Golden Scenarios & Frame Generator
Provides realistic synthetic computer vision frames and golden demo feeds.
Ensures the live hackathon pitch works 100% flawlessly without relying
on room lighting, awkward physical falls on stage, or external hardware glitches.
"""

from __future__ import annotations

import math
import time
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np


class GoldenScenarioGenerator:
    def __init__(self, width: int = 640, height: int = 480):
        self.width = width
        self.height = height
        self.step_counter = 0

    def generate_frame(
        self,
        scenario: str = "walk",
        t: float = 0.0,
    ) -> Tuple[np.ndarray, Tuple[int, int, int, int], Dict[str, Tuple[int, int]]]:
        """
        Renders a photorealistic dark tactical living room scene with an avatar skeleton.
        scenario: 'walk', 'sujud', 'fall'
        Returns: (frame_bgr, bbox_xywh, keypoints_dict)
        """
        self.step_counter += 1
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # Background: Modern living room interior (Dark Slate)
        # Floor plane
        floor_y = int(self.height * 0.72)
        cv2.rectangle(frame, (0, 0), (self.width, floor_y), (14, 18, 26), -1)  # Wall
        cv2.rectangle(frame, (0, floor_y), (self.width, self.height), (22, 28, 38), -1)  # Floor

        # Hairline Grid pattern on floor for perspective
        for x_line in range(0, self.width, 80):
            cv2.line(frame, (x_line, floor_y), (int((x_line - self.width / 2) * 1.5 + self.width / 2), self.height), (32, 40, 54), 1, cv2.LINE_AA)
        cv2.line(frame, (0, floor_y), (self.width, floor_y), (48, 58, 76), 1, cv2.LINE_AA)

        # Ambient Furniture: Couch outline on left, Lamp on right
        cv2.rectangle(frame, (40, floor_y - 70), (160, floor_y), (28, 35, 48), -1)
        cv2.rectangle(frame, (40, floor_y - 70), (160, floor_y), (45, 55, 75), 1)
        cv2.putText(frame, "Sofa", (80, floor_y - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (70, 85, 110), 1)

        # Camera Timestamp & Node Watermark (Top Left)
        time_str = time.strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, f"CAM-01 [LIVING ROOM] · {time_str} · 30 FPS", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 115, 135), 1, cv2.LINE_AA)
        cv2.putText(frame, "GUARDIANBOX EDGE CAPTURE (ZERO CLOUD EGRESS)", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (60, 180, 120), 1, cv2.LINE_AA)

        # Generate Avatar Pose based on Scenario
        cx = self.width // 2
        kpts: Dict[str, Tuple[int, int]] = {}

        if scenario == "walk":
            # Normal upright walking movement
            bob = int(math.sin(self.step_counter * 0.2) * 6)
            shift_x = int(math.sin(self.step_counter * 0.05) * 60)
            base_x = cx + shift_x
            base_y = floor_y - 40 + bob

            head_y = base_y - 140
            neck_y = head_y + 25
            hip_y = neck_y + 55
            knee_y = hip_y + 35
            foot_y = floor_y

            kpts = {
                "nose": (base_x, head_y),
                "neck": (base_x, neck_y),
                "left_shoulder": (base_x - 22, neck_y + 5),
                "right_shoulder": (base_x + 22, neck_y + 5),
                "left_elbow": (base_x - 30, neck_y + 30),
                "right_elbow": (base_x + 30, neck_y + 30),
                "hip": (base_x, hip_y),
                "left_knee": (base_x - 14, knee_y),
                "right_knee": (base_x + 14, knee_y),
                "left_foot": (base_x - 18, foot_y),
                "right_foot": (base_x + 18, foot_y),
            }
            # Upright bounding box (w < h, AR ~0.4)
            bbox = (base_x - 45, head_y - 15, 90, foot_y - head_y + 20)

        elif scenario == "sujud" or scenario == "prayer":
            # Muslim Prayer Mat (Sejadah) rendered on floor
            mat_x1, mat_y1 = cx - 110, floor_y - 15
            mat_x2, mat_y2 = cx + 90, floor_y + 45
            cv2.rectangle(frame, (mat_x1, mat_y1), (mat_x2, mat_y2), (20, 50, 40), -1)
            cv2.rectangle(frame, (mat_x1, mat_y1), (mat_x2, mat_y2), (30, 90, 70), 1)
            cv2.putText(frame, "Sejadah (Prayer Mat)", (mat_x1 + 10, mat_y2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (50, 140, 100), 1)

            # Controlled Sujud (Prostration) Geometry
            # Head and palms touch the floor forward, knees tucked underneath rear hip
            head_x = cx - 60
            head_y = floor_y + 10
            neck_x, neck_y = head_x + 30, floor_y - 15
            hip_x, hip_y = neck_x + 50, floor_y - 45  # Elevated hip
            knee_x, knee_y = hip_x + 15, floor_y + 10 # Knees on floor
            foot_x, foot_y = knee_x + 35, floor_y + 12

            kpts = {
                "nose": (head_x, head_y),
                "neck": (neck_x, neck_y),
                "left_shoulder": (neck_x, neck_y - 8),
                "right_shoulder": (neck_x, neck_y + 8),
                "left_elbow": (head_x + 15, head_y - 5),
                "right_elbow": (head_x + 15, head_y + 5),
                "hip": (hip_x, hip_y),
                "left_knee": (knee_x - 5, knee_y),
                "right_knee": (knee_x + 5, knee_y),
                "left_foot": (foot_x, foot_y),
                "right_foot": (foot_x + 10, foot_y),
            }
            # Prayer bounding box (w slightly > h, compact, AR ~1.3)
            bbox = (head_x - 20, hip_y - 15, 175, foot_y - hip_y + 35)

        elif scenario == "fall":
            # Acute Traumatic Fall
            # Person lying flat on bare floor, limbs splayed asymmetrically
            head_x = cx - 110
            head_y = floor_y + 20
            neck_x, neck_y = head_x + 40, floor_y + 18
            hip_x, hip_y = neck_x + 85, floor_y + 22
            knee_x, knee_y = hip_x + 65, floor_y + 15
            foot_x, foot_y = knee_x + 60, floor_y + 28

            kpts = {
                "nose": (head_x, head_y),
                "neck": (neck_x, neck_y),
                "left_shoulder": (neck_x - 10, neck_y - 18),
                "right_shoulder": (neck_x + 10, neck_y + 18),
                "left_elbow": (neck_x - 25, neck_y - 35),  # Thrown arm
                "right_elbow": (neck_x + 30, neck_y + 30),
                "hip": (hip_x, hip_y),
                "left_knee": (knee_x - 10, knee_y - 20),
                "right_knee": (knee_x + 10, knee_y + 15),
                "left_foot": (foot_x - 15, foot_y - 15),
                "right_foot": (foot_x, foot_y + 10),
            }
            # Extreme horizontal bounding box (w >> h, AR ~2.6)
            bbox = (head_x - 25, floor_y - 25, 290, 85)
        else:
            bbox = (cx - 40, floor_y - 120, 80, 120)

        # Draw subtle shaded silhouette behind skeleton
        x, y, w, h = bbox
        overlay = frame.copy()
        cv2.ellipse(overlay, (x + w // 2, y + h // 2), (w // 2, h // 2), 0, 0, 360, (35, 45, 60), -1)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

        return frame, bbox, kpts
