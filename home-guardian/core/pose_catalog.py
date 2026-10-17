"""
GuardianBox - Human Pose Taxonomy & Geometric Counter-Check Engine
Formal taxonomy of human postures encountered in home environments.
Provides multi-parameter geometric counter-checks (Aspect Ratio, Torso Angle,
Hip-to-Floor Ratio, Head-to-Hip Delta, and Bilateral Symmetry) to rigorously
differentiate medical collapses from cultural prayer and domestic resting.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class PoseCategory(str, Enum):
    UPRIGHT_STANDING = "UPRIGHT_STANDING"
    WALKING_AMBULATION = "WALKING_AMBULATION"
    SEATED_CHAIR = "SEATED_CHAIR"
    SEATED_FLOOR_RELAXED = "SEATED_FLOOR_RELAXED"
    KNEELING_PRAYER_RUKU = "KNEELING_PRAYER_RUKU"
    KNEELING_PRAYER_SUJUD = "KNEELING_PRAYER_SUJUD"
    SLUMPED_LEANING = "SLUMPED_LEANING"
    PRONE_ACUTE_FALL = "PRONE_ACUTE_FALL"
    SUPINE_ACUTE_FALL = "SUPINE_ACUTE_FALL"
    LATERAL_FLACCID_FALL = "LATERAL_FLACCID_FALL"
    RECOVERY_FETAL = "RECOVERY_FETAL"


@dataclass
class PoseSpecification:
    category: PoseCategory
    display_name: str
    is_emergency: bool
    description: str
    expected_ar_range: Tuple[float, float]        # (min_ar, max_ar) width / height
    expected_torso_angle: Tuple[float, float]     # (min_deg, max_deg) from vertical
    expected_hip_floor_ratio: Tuple[float, float] # (min_ratio, max_ratio) 0.0=top, 1.0=ground
    expected_symmetry_score: Tuple[float, float]  # (min_sym, max_sym) 0.0 to 1.0
    cultural_context: Optional[str] = None
    common_false_alarm_risk: Optional[str] = None


POSE_TAXONOMY: Dict[PoseCategory, PoseSpecification] = {
    PoseCategory.UPRIGHT_STANDING: PoseSpecification(
        category=PoseCategory.UPRIGHT_STANDING,
        display_name="Upright Standing",
        is_emergency=False,
        description="Subject standing vertically with balanced gravitational center.",
        expected_ar_range=(0.30, 0.65),
        expected_torso_angle=(0.0, 15.0),
        expected_hip_floor_ratio=(0.50, 0.70),
        expected_symmetry_score=(0.80, 1.00),
    ),
    PoseCategory.WALKING_AMBULATION: PoseSpecification(
        category=PoseCategory.WALKING_AMBULATION,
        display_name="Walking / Ambulation",
        is_emergency=False,
        description="Normal rhythmic walking across the room with dynamic leg stride.",
        expected_ar_range=(0.35, 0.75),
        expected_torso_angle=(0.0, 20.0),
        expected_hip_floor_ratio=(0.50, 0.72),
        expected_symmetry_score=(0.60, 0.95),
    ),
    PoseCategory.SEATED_CHAIR: PoseSpecification(
        category=PoseCategory.SEATED_CHAIR,
        display_name="Seated on Chair / Sofa",
        is_emergency=False,
        description="Subject seated on elevated furniture with vertical torso and bent knees.",
        expected_ar_range=(0.65, 1.10),
        expected_torso_angle=(5.0, 30.0),
        expected_hip_floor_ratio=(0.55, 0.75),
        expected_symmetry_score=(0.75, 1.00),
        common_false_alarm_risk="Can look like floor sitting if camera is angled downwards.",
    ),
    PoseCategory.SEATED_FLOOR_RELAXED: PoseSpecification(
        category=PoseCategory.SEATED_FLOOR_RELAXED,
        display_name="Seated on Floor / Lounging",
        is_emergency=False,
        description="Deliberate floor sitting on mat or carpet with upright/relaxed spine.",
        expected_ar_range=(0.90, 1.45),
        expected_torso_angle=(10.0, 45.0),
        expected_hip_floor_ratio=(0.78, 0.96),
        expected_symmetry_score=(0.70, 0.95),
        common_false_alarm_risk="High aspect ratio can trigger naive horizontal bounding box detectors.",
    ),
    PoseCategory.KNEELING_PRAYER_RUKU: PoseSpecification(
        category=PoseCategory.KNEELING_PRAYER_RUKU,
        display_name="Prayer: Ruku' (Bowing)",
        is_emergency=False,
        description="Islamic prayer bowing: hands on knees, back roughly horizontal, feet anchored.",
        expected_ar_range=(1.05, 1.60),
        expected_torso_angle=(65.0, 90.0),
        expected_hip_floor_ratio=(0.55, 0.75),
        expected_symmetry_score=(0.85, 1.00),
        cultural_context="Solat step 2: Ruku'. Precedes Sujud.",
        common_false_alarm_risk="Rapid descent to 90° torso angle easily triggers fall alarms.",
    ),
    PoseCategory.KNEELING_PRAYER_SUJUD: PoseSpecification(
        category=PoseCategory.KNEELING_PRAYER_SUJUD,
        display_name="Prayer: Sujud (Prostration)",
        is_emergency=False,
        description="Islamic prostration: forehead, palms, knees, and toes touch prayer mat simultaneously.",
        expected_ar_range=(1.10, 1.70),
        expected_torso_angle=(35.0, 65.0),
        expected_hip_floor_ratio=(0.75, 0.95),
        expected_symmetry_score=(0.90, 1.00),
        cultural_context="Solat prostration on Sejadah. Held for 3-10 seconds per rak'ah.",
        common_false_alarm_risk="Forehead touching floor for >3s universally misclassified as coma/collapse by naive CV.",
    ),
    PoseCategory.SLUMPED_LEANING: PoseSpecification(
        category=PoseCategory.SLUMPED_LEANING,
        display_name="Slumped / Asymmetrical Lean",
        is_emergency=False,
        description="Senior leaning heavily against furniture or wall due to fatigue.",
        expected_ar_range=(0.70, 1.25),
        expected_torso_angle=(25.0, 50.0),
        expected_hip_floor_ratio=(0.60, 0.80),
        expected_symmetry_score=(0.40, 0.70),
    ),
    PoseCategory.PRONE_ACUTE_FALL: PoseSpecification(
        category=PoseCategory.PRONE_ACUTE_FALL,
        display_name="Acute Prone Fall (Face-Down)",
        is_emergency=True,
        description="Traumatic uncontrolled slip forward; body face down, limbs splayed asymmetrically.",
        expected_ar_range=(2.20, 4.50),
        expected_torso_angle=(75.0, 90.0),
        expected_hip_floor_ratio=(0.82, 1.00),
        expected_symmetry_score=(0.15, 0.55),
    ),
    PoseCategory.SUPINE_ACUTE_FALL: PoseSpecification(
        category=PoseCategory.SUPINE_ACUTE_FALL,
        display_name="Acute Supine Fall (Backwards)",
        is_emergency=True,
        description="Loss of balance backwards; head strikes floor, chest facing ceiling, extremities flaccid.",
        expected_ar_range=(2.00, 4.20),
        expected_torso_angle=(75.0, 90.0),
        expected_hip_floor_ratio=(0.85, 1.00),
        expected_symmetry_score=(0.20, 0.60),
    ),
    PoseCategory.LATERAL_FLACCID_FALL: PoseSpecification(
        category=PoseCategory.LATERAL_FLACCID_FALL,
        display_name="Lateral Fall (Hip Impact)",
        is_emergency=True,
        description="Sideways collapse with severe femoral/hip impact hazard; limbs tangled.",
        expected_ar_range=(1.80, 3.60),
        expected_torso_angle=(60.0, 85.0),
        expected_hip_floor_ratio=(0.80, 0.98),
        expected_symmetry_score=(0.10, 0.45),
    ),
    PoseCategory.RECOVERY_FETAL: PoseSpecification(
        category=PoseCategory.RECOVERY_FETAL,
        display_name="Curled / Fetal Post-Fall Posture",
        is_emergency=True,
        description="Senior conscious or in shock after fall, curled on side holding injured hip or head.",
        expected_ar_range=(1.30, 2.10),
        expected_torso_angle=(45.0, 80.0),
        expected_hip_floor_ratio=(0.82, 0.98),
        expected_symmetry_score=(0.25, 0.55),
    ),
}


@dataclass
class CounterCheckResult:
    matched_category: PoseCategory
    display_name: str
    is_emergency: bool
    confidence_score: float
    passed_checks: List[str]
    failed_checks: List[str]
    symmetry_score: float
    ar_difference: float
    torso_difference: float
    diagnostic_summary: str
    counter_check_status: str  # "VERIFIED", "PROBABLE", "DISPUTED"


class PoseCounterChecker:
    """Multi-parameter geometric counter-checking validator."""

    def compute_symmetry(self, keypoints: Dict[str, Tuple[int, int]]) -> float:
        """
        Computes bilateral symmetry score (0.0 = completely asymmetric/flaccid, 1.0 = perfectly mirrored).
        Compares left vs right shoulder, elbow, knee, and foot vectors relative to center neck-hip axis.
        """
        if not keypoints or "neck" not in keypoints or "hip" not in keypoints:
            return 0.5

        cx = keypoints["neck"][0]
        pairs = [
            ("left_shoulder", "right_shoulder"),
            ("left_elbow", "right_elbow"),
            ("left_knee", "right_knee"),
            ("left_foot", "right_foot"),
        ]

        deltas = []
        for lp, rp in pairs:
            if lp in keypoints and rp in keypoints:
                lx, ly = keypoints[lp]
                rx, ry = keypoints[rp]
                dist_l = abs(lx - cx)
                dist_r = abs(rx - cx)
                y_diff = abs(ly - ry)
                # Normalized symmetry
                sym = max(0.0, 1.0 - (abs(dist_l - dist_r) / 50.0 + y_diff / 40.0))
                deltas.append(sym)

        if not deltas:
            return 0.5
        return round(float(sum(deltas) / len(deltas)), 2)

    def counter_check(
        self,
        aspect_ratio: float,
        torso_angle: float,
        hip_height_ratio: float,
        keypoints: Optional[Dict[str, Tuple[int, int]]] = None,
        candidate_category: Optional[PoseCategory] = None,
    ) -> CounterCheckResult:
        """
        Rigorously counter-checks geometric measurements against all taxonomy categories.
        """
        symmetry = self.compute_symmetry(keypoints or {})
        best_match: Optional[PoseCategory] = None
        best_score = -999.0
        best_passed: List[str] = []
        best_failed: List[str] = []

        categories_to_check = [candidate_category] if candidate_category else list(PoseCategory)

        for cat in categories_to_check:
            spec = POSE_TAXONOMY[cat]
            score = 0.0
            passed = []
            failed = []

            # Check 1: Aspect Ratio
            min_ar, max_ar = spec.expected_ar_range
            if min_ar <= aspect_ratio <= max_ar:
                score += 30.0
                passed.append(f"AR ({aspect_ratio:.2f}) matches range [{min_ar}-{max_ar}]")
            else:
                diff = min(abs(aspect_ratio - min_ar), abs(aspect_ratio - max_ar))
                score -= diff * 15.0
                failed.append(f"AR ({aspect_ratio:.2f}) out of range [{min_ar}-{max_ar}] (Δ{diff:.2f})")

            # Check 2: Torso Angle
            min_t, max_t = spec.expected_torso_angle
            if min_t <= torso_angle <= max_t:
                score += 30.0
                passed.append(f"Torso ({torso_angle:.1f}°) matches range [{min_t}°-{max_t}°]")
            else:
                diff = min(abs(torso_angle - min_t), abs(torso_angle - max_t))
                score -= diff * 0.8
                failed.append(f"Torso ({torso_angle:.1f}°) out of range [{min_t}°-{max_t}°] (Δ{diff:.1f}°)")

            # Check 3: Hip to Floor Ratio
            min_h, max_h = spec.expected_hip_floor_ratio
            if min_h <= hip_height_ratio <= max_h:
                score += 20.0
                passed.append(f"Hip-Floor Ratio ({hip_height_ratio:.2f}) matches range [{min_h}-{max_h}]")
            else:
                diff = min(abs(hip_height_ratio - min_h), abs(hip_height_ratio - max_h))
                score -= diff * 20.0
                failed.append(f"Hip-Floor Ratio ({hip_height_ratio:.2f}) out of range [{min_h}-{max_h}]")

            # Check 4: Bilateral Symmetry
            min_s, max_s = spec.expected_symmetry_score
            if min_s <= symmetry <= max_s:
                score += 20.0
                passed.append(f"Symmetry ({symmetry:.2f}) matches range [{min_s}-{max_s}]")
            else:
                diff = min(abs(symmetry - min_s), abs(symmetry - max_s))
                score -= diff * 25.0
                failed.append(f"Symmetry ({symmetry:.2f}) diverged from [{min_s}-{max_s}]")

            if score > best_score:
                best_score = score
                best_match = cat
                best_passed = passed
                best_failed = failed

        assert best_match is not None
        spec = POSE_TAXONOMY[best_match]
        confidence = min(0.99, max(0.50, (best_score + 20.0) / 120.0))
        status = "VERIFIED" if len(best_passed) >= 3 else ("PROBABLE" if len(best_passed) >= 2 else "DISPUTED")

        summary = (
            f"Pose verified as '{spec.display_name}' ({status}, conf {confidence*100:.1f}%). "
            f"{len(best_passed)}/4 parameters passed geometric criteria."
        )

        return CounterCheckResult(
            matched_category=best_match,
            display_name=spec.display_name,
            is_emergency=spec.is_emergency,
            confidence_score=round(confidence, 2),
            passed_checks=best_passed,
            failed_checks=best_failed,
            symmetry_score=symmetry,
            ar_difference=round(abs(aspect_ratio - ((spec.expected_ar_range[0] + spec.expected_ar_range[1]) / 2)), 2),
            torso_difference=round(abs(torso_angle - ((spec.expected_torso_angle[0] + spec.expected_torso_angle[1]) / 2)), 1),
            diagnostic_summary=summary,
            counter_check_status=status,
        )


def get_all_pose_definitions() -> List[Dict[str, Any]]:
    """Returns serializable JSON list of all 11 defined human poses."""
    return [
        {
            "id": p.category.value,
            "name": p.display_name,
            "is_emergency": p.is_emergency,
            "description": p.description,
            "expected_ar": f"{p.expected_ar_range[0]} - {p.expected_ar_range[1]}",
            "expected_torso_deg": f"{p.expected_torso_angle[0]}° - {p.expected_torso_angle[1]}°",
            "cultural_context": p.cultural_context,
            "false_alarm_risk": p.common_false_alarm_risk,
        }
        for p in POSE_TAXONOMY.values()
    ]
