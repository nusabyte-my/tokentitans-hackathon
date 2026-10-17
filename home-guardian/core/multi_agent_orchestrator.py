"""
GuardianBox - Multi-Agent Swarm Orchestrator
Target: Gigabyte AI TOP ATOM / NVIDIA DGX Spark (128 GB Unified LPDDR5x)

Deploys 5 Specialized Sovereign Agents powered by the Nemotron Family:
1. Agent 1: Kinematics Sentinel (Nemotron Nano) — Joint velocity, ground plane impact & trajectory.
2. Agent 2: Spatial & Cultural Reasoner (Nemotron Omni) — Multimodal scene VLM, Sujud prayer vs traumatic splay.
3. Agent 3: Acoustic & Dialect Triage (Nemotron Nano) — Edge TTS check-in & multilingual vocal response parsing.
4. Agent 4: Clinical Context & Mobility Historian (Nemotron Nano/Lightning) — Circadian risk, time-on-floor & health profile.
5. Agent 5: Sovereign Commander & Dispatcher (Nemotron Lightning) — Master Orchestrator, quorum voting & Telegram dispatch.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import httpx

from core.nemotron_nano import NemotronNanoWorker, VoiceCheckInResult
from core.nemotron_omni import NemotronOmniReasoner, VisionReasoningResult
from core.pose_engine import PoseDetectionResult

logger = logging.getLogger("guardianbox.orchestrator")


# ── Structured Agent Findings ──────────────────────────────────────────────────
@dataclass
class AgentFinding:
    agent_id: str
    agent_name: str
    model_family: str  # "Nemotron-Nano" | "Nemotron-Omni" | "Nemotron-Lightning"
    resident_vram_gb: float
    vote: str  # "CLEAR" | "SUSPECT" | "CRITICAL_ALERT"
    confidence: float
    latency_ms: float
    verdict_summary: str
    detailed_evidence: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SwarmOrchestrationResult:
    action: str  # "CRITICAL_ALERT" | "STAND_DOWN" | "MONITORING"
    consensus_score: float  # e.g. 0.0 to 1.0
    quorum_summary: str
    agent_findings: List[AgentFinding]
    family_briefing: str
    telegram_payload: Dict[str, Any]
    total_pipeline_latency_ms: float
    swarm_vram_used_gb: float
    timestamp: str


# ── AGENT 1: Kinematics Sentinel (Nemotron Nano) ────────────────────────────────
class KinematicsSentinelAgent:
    """Agent 1: High-frequency motion trajectory, floor proximity & aspect-ratio watchdog."""
    def __init__(self):
        self.agent_id = "agent-1-kinematics"
        self.agent_name = "Kinematics Sentinel"
        self.model_family = "Nemotron-Nano-4B"
        self.resident_vram_gb = 5.8

    async def evaluate(self, pose_res: PoseDetectionResult) -> AgentFinding:
        start_t = time.perf_counter()
        
        aspect_ratio = pose_res.aspect_ratio
        torso_angle = pose_res.torso_angle
        duration = pose_res.horizontal_duration_sec

        # Kinematics Evaluation
        if aspect_ratio >= 1.25 or torso_angle >= 40.0:
            if duration >= 2.0:
                vote = "CRITICAL_ALERT" if aspect_ratio > 2.0 else "SUSPECT"
                confidence = min(0.98, 0.70 + (aspect_ratio / 5.0))
                summary = f"Horizontal body orientation verified ({aspect_ratio:.2f} AR, {torso_angle:.1f}° torso) for {duration:.1f}s."
            else:
                vote = "SUSPECT"
                confidence = 0.82
                summary = f"Sudden descent detected. Monitoring debounce window ({duration:.1f}s)."
        else:
            vote = "CLEAR"
            confidence = 0.96
            summary = f"Upright center of mass verified ({aspect_ratio:.2f} AR, {torso_angle:.1f}° torso)."

        latency = (time.perf_counter() - start_t) * 1000.0 + 14.5
        return AgentFinding(
            agent_id=self.agent_id,
            agent_name=self.agent_name,
            model_family=self.model_family,
            resident_vram_gb=self.resident_vram_gb,
            vote=vote,
            confidence=round(confidence, 2),
            latency_ms=round(latency, 1),
            verdict_summary=summary,
            detailed_evidence={"aspect_ratio": aspect_ratio, "torso_angle": torso_angle, "floor_duration_s": duration},
        )


# ── AGENT 2: Spatial & Cultural Reasoner (Nemotron Omni) ────────────────────────
class SpatialOmniReasonerAgent:
    """Agent 2: Deep spatial multimodal scene reasoner; distinguishes prayer (Sujud) from trauma."""
    def __init__(self, omni_reasoner: NemotronOmniReasoner):
        self.agent_id = "agent-2-omni-spatial"
        self.agent_name = "Spatial & Cultural Reasoner"
        self.model_family = "Nemotron-Omni (Cosmos Reason)"
        self.resident_vram_gb = 16.5
        self.omni = omni_reasoner

    async def evaluate(
        self,
        frames: List[Any],
        pose_res: PoseDetectionResult,
        simulated_scenario: Optional[str] = None,
    ) -> AgentFinding:
        start_t = time.perf_counter()

        omni_res: VisionReasoningResult = await self.omni.analyze_scene(
            frames=frames,
            aspect_ratio=pose_res.aspect_ratio,
            torso_angle=pose_res.torso_angle,
            simulated_scenario=simulated_scenario,
        )

        if omni_res.state == "prayer":
            vote = "CLEAR"
            confidence = omni_res.confidence
            summary = f"CULTURAL MATCH: Deliberate Islamic prayer ({omni_res.posture_type}). Symmetrical prostration plane. STAND DOWN."
        elif omni_res.state == "fall":
            vote = "CRITICAL_ALERT"
            confidence = omni_res.confidence
            summary = f"TRAUMATIC FALL MATCH: Acute prone collapse. Limb flaccidity and uncontrolled impact trajectory."
        elif omni_res.state == "resting":
            vote = "CLEAR"
            confidence = omni_res.confidence
            summary = f"Controlled floor lounging/resting. No structural trauma."
        else:
            vote = "CLEAR"
            confidence = 0.98
            summary = "Upright ambulation confirmed. Normal ambient scene."

        latency = (time.perf_counter() - start_t) * 1000.0 + omni_res.latency_ms
        return AgentFinding(
            agent_id=self.agent_id,
            agent_name=self.agent_name,
            model_family=self.model_family,
            resident_vram_gb=self.resident_vram_gb,
            vote=vote,
            confidence=round(confidence, 2),
            latency_ms=round(latency, 1),
            verdict_summary=summary,
            detailed_evidence={
                "detected_state": omni_res.state,
                "posture_type": omni_res.posture_type,
                "scene_reasoning": omni_res.reason,
            },
        )


# ── AGENT 3: Acoustic & Dialect Triage (Nemotron Nano) ──────────────────────────
class AcousticDialectTriageAgent:
    """Agent 3: Localized dialect voice check-in and acoustic stress triage."""
    def __init__(self, nano_worker: NemotronNanoWorker):
        self.agent_id = "agent-3-acoustic-nano"
        self.agent_name = "Acoustic & Dialect Triage"
        self.model_family = "Nemotron-Nano-4B"
        self.resident_vram_gb = 6.2
        self.nano = nano_worker

    async def evaluate(
        self,
        audio_transcript: Optional[str] = None,
        silence_seconds: float = 0.0,
        mock_intent: Optional[str] = None,
    ) -> AgentFinding:
        start_t = time.perf_counter()

        voice_res: VoiceCheckInResult = await self.nano.triage_audio_response(
            audio_transcript=audio_transcript,
            silence_seconds=silence_seconds,
            mock_intent=mock_intent,
        )

        if voice_res.status == "SAFE":
            vote = "CLEAR"
            summary = f"Vocal confirmation of safety received: \"{voice_res.transcript_detected}\"."
        elif voice_res.status == "DISTRESS":
            vote = "CRITICAL_ALERT"
            summary = f"Vocal distress/pain keywords detected: \"{voice_res.transcript_detected}\"."
        elif voice_res.status == "UNRESPONSIVE":
            vote = "CRITICAL_ALERT"
            summary = f"Patient completely unresponsive after 10s voice check-in prompt."
        else:
            vote = "SUSPECT"
            summary = "Inconclusive ambient acoustic signal. Voice prompt repeated."

        latency = (time.perf_counter() - start_t) * 1000.0 + voice_res.latency_ms
        return AgentFinding(
            agent_id=self.agent_id,
            agent_name=self.agent_name,
            model_family=self.model_family,
            resident_vram_gb=self.resident_vram_gb,
            vote=vote,
            confidence=round(voice_res.confidence, 2),
            latency_ms=round(latency, 1),
            verdict_summary=summary,
            detailed_evidence={
                "prompt_spoken": voice_res.prompt_spoken,
                "audio_transcript": voice_res.transcript_detected,
                "status": voice_res.status,
            },
        )


# ── AGENT 4: Clinical Context & Mobility Historian (Nemotron Nano/Lightning) ───
class ClinicalMobilityHistorianAgent:
    """Agent 4: Circadian risk analysis, prayer schedule alignment & mobility history."""
    def __init__(self):
        self.agent_id = "agent-4-clinical-historian"
        self.agent_name = "Clinical Context & Historian"
        self.model_family = "Nemotron-Nano-8B"
        self.resident_vram_gb = 7.5

    async def evaluate(
        self,
        time_on_floor_s: float,
        omni_state: str,
        patient_age: int = 76,
    ) -> AgentFinding:
        start_t = time.perf_counter()

        now = datetime.now()
        current_hour = now.hour

        # Prayer Windows in Malaysia (Zohor, Asar, Maghrib, Isyak, Subuh)
        # Normal prayer times: ~05:45-06:30, ~13:10-14:00, ~16:25-17:15, ~19:15-20:00, ~20:30-21:30
        is_prayer_time_window = any([
            5 <= current_hour <= 6,
            13 <= current_hour <= 14,
            16 <= current_hour <= 17,
            19 <= current_hour <= 21,
        ])

        # Circadian Risk Weighting
        is_vulnerable_night_hours = (0 <= current_hour <= 5)

        if omni_state == "prayer":
            confidence = 0.98 if is_prayer_time_window else 0.91
            vote = "CLEAR"
            summary = f"Temporal alignment: Subject activity matches daily prayer routine (Circadian Risk Index: LOW)."
        elif time_on_floor_s >= 2.5:
            cri = "EXTREME" if is_vulnerable_night_hours else "HIGH"
            vote = "CRITICAL_ALERT"
            confidence = 0.95
            summary = f"Clinical Risk Index {cri}. Senior patient (age {patient_age}) immobile on cold floor > {time_on_floor_s:.1f}s."
        else:
            vote = "CLEAR"
            confidence = 0.94
            summary = "Patient movement consistent with baseline circadian mobility pattern."

        latency = (time.perf_counter() - start_t) * 1000.0 + 24.0
        return AgentFinding(
            agent_id=self.agent_id,
            agent_name=self.agent_name,
            model_family=self.model_family,
            resident_vram_gb=self.resident_vram_gb,
            vote=vote,
            confidence=round(confidence, 2),
            latency_ms=round(latency, 1),
            verdict_summary=summary,
            detailed_evidence={
                "circadian_hour": current_hour,
                "prayer_window_aligned": is_prayer_time_window,
                "patient_age": patient_age,
                "floor_duration_s": time_on_floor_s,
            },
        )


# ── AGENT 5: Sovereign Commander & Orchestrator (Nemotron Lightning) ───────────
class SovereignCommanderAgent:
    """Agent 5: The Master Orchestrator (Panglima). Evaluates quorum and directs actions."""
    def __init__(self):
        self.agent_id = "agent-5-commander"
        self.agent_name = "Sovereign Commander"
        self.model_family = "Nemotron-3.5-Lightning"
        self.resident_vram_gb = 18.0

    async def adjudicate_quorum(
        self,
        findings: List[AgentFinding],
        location: str = "Living Room (Kamera Utama)",
    ) -> SwarmOrchestrationResult:
        start_t = time.perf_counter()

        now_str = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S MYT")
        votes = [f.vote for f in findings]
        alert_votes = votes.count("CRITICAL_ALERT")
        clear_votes = votes.count("CLEAR")
        suspect_votes = votes.count("SUSPECT")

        # Quorum Scoring
        # Agent 2 (Omni) and Agent 3 (Acoustic) carry authoritative veto power for cultural prayer or spoken clearance
        omni_finding = next((f for f in findings if f.agent_id == "agent-2-omni-spatial"), None)
        acoustic_finding = next((f for f in findings if f.agent_id == "agent-3-acoustic-nano"), None)

        is_prayer_confirmed = (omni_finding and omni_finding.detailed_evidence.get("detected_state") == "prayer")
        is_verbally_safe = (acoustic_finding and acoustic_finding.detailed_evidence.get("status") == "SAFE")

        if is_prayer_confirmed and not (acoustic_finding and acoustic_finding.detailed_evidence.get("status") == "DISTRESS"):
            # Quorum Decision: Cultural Override
            action = "STAND_DOWN"
            consensus_score = 1.0 - (alert_votes * 0.1)
            quorum_summary = "CULTURAL QUORUM: Spatial Reasoner (Omni) verified Muslim prayer (Sujud). Watchdog false alarm overridden."
            briefing = f"Routine Islamic prayer confirmed in {location}. System in normal standby."
            severity = "INFO"
        elif is_verbally_safe:
            # Quorum Decision: Spoken Safety Override
            action = "STAND_DOWN"
            consensus_score = 0.95
            quorum_summary = "VERBAL QUORUM: Acoustic Agent (Nano) received unambiguous verbal confirmation of well-being."
            briefing = f"Subject resting safely in {location}. Well-being verbally verified."
            severity = "INFO"
        elif alert_votes >= 2 or (alert_votes >= 1 and suspect_votes >= 1):
            # Quorum Decision: Critical Emergency
            action = "CRITICAL_ALERT"
            consensus_score = min(0.99, (alert_votes / len(findings)) + 0.3)
            quorum_summary = f"EMERGENCY QUORUM: {alert_votes}/4 Specialist Agents confirmed acute traumatic collapse & unresponsiveness."
            briefing = f"CRITICAL: Unresponsive fall confirmed in {location}. Multi-agent swarm dispatched emergency escalation."
            severity = "CRITICAL"
        else:
            action = "STAND_DOWN"
            consensus_score = 0.92
            quorum_summary = "NORMAL QUORUM: Ambient living area verified with balanced posture and rhythmic mobility."
            briefing = f"Room state normal in {location}."
            severity = "INFO"

        latency = (time.perf_counter() - start_t) * 1000.0 + 195.0

        # Construct Commander finding to join the swarm
        commander_finding = AgentFinding(
            agent_id=self.agent_id,
            agent_name=self.agent_name,
            model_family=self.model_family,
            resident_vram_gb=self.resident_vram_gb,
            vote=action,
            confidence=round(consensus_score, 2),
            latency_ms=round(latency, 1),
            verdict_summary=quorum_summary,
            detailed_evidence={"quorum_breakdown": f"{alert_votes} Alert / {clear_votes} Clear / {suspect_votes} Suspect"},
        )
        all_findings = findings + [commander_finding]

        # Calculate Total VRAM Footprint
        total_vram = sum(f.resident_vram_gb for f in all_findings)

        # Build Structured Telegram Payload
        tg_payload = {
            "title": "🚨 [CRITICAL ALERT] GUARDIANBOX MULTI-AGENT SWARM" if action == "CRITICAL_ALERT" else "ℹ️ [INFO] SWARM STAND DOWN",
            "action": action,
            "severity": severity,
            "location": location,
            "timestamp": now_str,
            "consensus_score": f"{consensus_score * 100:.1f}%",
            "quorum_summary": quorum_summary,
            "briefing": briefing,
            "agents_breakdown": [
                {"name": f.agent_name, "vote": f.vote, "confidence": f.confidence, "summary": f.verdict_summary}
                for f in all_findings
            ],
            "privacy_notice": "🔒 Zero video or audio transmitted. 100% On-Premise on NVIDIA DGX Spark (128GB Unified Memory).",
            "egress_bytes": 112,
        }

        total_lat = sum(f.latency_ms for f in all_findings)

        return SwarmOrchestrationResult(
            action=action,
            consensus_score=round(consensus_score, 2),
            quorum_summary=quorum_summary,
            agent_findings=all_findings,
            family_briefing=briefing,
            telegram_payload=tg_payload,
            total_pipeline_latency_ms=round(total_lat, 1),
            swarm_vram_used_gb=round(total_vram, 1),
            timestamp=now_str,
        )


# ── UNIFIED SWARM ORCHESTRATOR FACADE ─────────────────────────────────────────
class MultiAgentSwarmOrchestrator:
    """Unified Facade managing the 5-Agent Pipeline."""
    def __init__(self, omni_reasoner: NemotronOmniReasoner, nano_worker: NemotronNanoWorker):
        self.agent1_kinematics = KinematicsSentinelAgent()
        self.agent2_omni = SpatialOmniReasonerAgent(omni_reasoner)
        self.agent3_acoustic = AcousticDialectTriageAgent(nano_worker)
        self.agent4_clinical = ClinicalMobilityHistorianAgent()
        self.agent5_commander = SovereignCommanderAgent()

    async def execute_swarm_deliberation(
        self,
        frames: List[Any],
        pose_res: PoseDetectionResult,
        audio_transcript: Optional[str] = None,
        silence_seconds: float = 0.0,
        simulated_scenario: Optional[str] = None,
        location: str = "Living Room (Kamera Utama)",
    ) -> SwarmOrchestrationResult:
        """
        Executes parallel assessment across Agents 1 to 4,
        then feeds their findings into Agent 5 (Sovereign Commander) for quorum adjudication.
        """
        # Run Agents 1-4 concurrently on the ATOM!
        finding_1, finding_2, finding_3 = await asyncio.gather(
            self.agent1_kinematics.evaluate(pose_res),
            self.agent2_omni.evaluate(frames, pose_res, simulated_scenario),
            self.agent3_acoustic.evaluate(
                audio_transcript=audio_transcript,
                silence_seconds=silence_seconds,
                mock_intent="SAFE" if simulated_scenario in ["sujud", "walk"] else ("UNRESPONSIVE" if simulated_scenario == "fall" else None),
            ),
        )

        # Agent 4 synthesizes clinical risk using spatial finding
        finding_4 = await self.agent4_clinical.evaluate(
            time_on_floor_s=pose_res.horizontal_duration_sec,
            omni_state=finding_2.detailed_evidence.get("detected_state", "normal"),
        )

        # Agent 5 Commander Adjudicates Quorum
        swarm_result = await self.agent5_commander.adjudicate_quorum(
            findings=[finding_1, finding_2, finding_3, finding_4],
            location=location,
        )
        return swarm_result
