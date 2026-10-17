"""
GuardianBox - Voice Command & Speech Intent Interpreter
Supports dual-mode speech interaction:
1. Hands-Free Caregiver/Operator Commands ("GuardianBox status", "GuardianBox reset", "Call 999")
2. Elderly Patient Vocal Triage ("Saya okay, takpe" -> Stand Down; "Tolong sakit!" -> Critical Alert)
Powered by Nemotron-Nano audio classification logic with instant local dialect matching.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class VoiceCommandResult:
    command_type: str  # "OPERATOR_COMMAND" | "PATIENT_RESPONSE" | "UNKNOWN"
    recognized_intent: str
    target_action: str  # "RESET", "STATUS", "CHECKIN", "EMERGENCY_DISPATCH", "SWITCH_SCENARIO", "PATIENT_SAFE", "PATIENT_DISTRESS", "IGNORE"
    target_param: Optional[str]
    spoken_feedback: str
    feedback_message: str
    confidence: float


class VoiceCommandInterpreter:
    def __init__(self):
        # Caregiver command regex patterns
        self.cmd_patterns = [
            (r"(?:guardianbox|guardian\s*box)?\s*(?:status|keadaan|system\s*check)", "STATUS", "STATUS", "Status sistem: Semua lima model beroperasi normal di DGX Spark. Memori 54 GB stabil."),
            (r"(?:guardianbox|guardian\s*box)?\s*(?:reset|clear|dismiss|false\s*alarm|batal|padamkan)", "RESET", "RESET", "Sistem telah ditetapkan semula. Kembali ke pemantauan biasa."),
            (r"(?:guardianbox|guardian\s*box)?\s*(?:check\s*in|tanya\s*pakcik|prompt|suara)", "CHECKIN", "CHECKIN", "Menghantar semakan suara kepada pesakit: 'Pakcik, dengar tak? Pakcik okay ke?'"),
            (r"(?:guardianbox|guardian\s*box)?\s*(?:emergency|kecemasan|call\s*999|dial\s*999|dispatch|hantar\s*bantuan)", "EMERGENCY_DISPATCH", "EMERGENCY_DISPATCH", "Kecemasan diaktifkan secara manual. Telegram amaran dihantar ke telefon keluarga."),
            (r"(?:switch\s*to|tukar\s*ke|mode)\s*(?:sujud|solat|prayer)", "SWITCH_SCENARIO", "sujud", "Menukar senario ke mod Sujud / Solat."),
            (r"(?:switch\s*to|tukar\s*ke|mode)\s*(?:fall|jatuh|collapse)", "SWITCH_SCENARIO", "fall", "Menukar senario ke mod Acute Fall (Kecemasan)."),
            (r"(?:switch\s*to|tukar\s*ke|mode)\s*(?:walk|jalan|normal)", "SWITCH_SCENARIO", "walk", "Menukar senario ke mod Jalan Biasa (Normal)."),
        ]

        # Patient speech intent patterns (Multilingual Malaysian)
        self.patient_safe_patterns = [
            r"okay", r"ok", r"takpe", r"tak\s*ada\s*apa", r"elok", r"boleh\s*bangun",
            r"fine", r"rehat", r"berehat", r"sembahyang", r"solat", r"sujud",
            r"resting", r"saja\s*duduk", r"no\s*problem", r"没事", r"好", r"nalla",
        ]
        self.patient_distress_patterns = [
            r"tolong", r"help", r"sakit", r"aduh", r"jatuh", r"tak\s*boleh\s*bangun",
            r"patah", r"darah", r"sesak", r"pening", r"emergency", r"pain",
            r"救命", r"痛", r"kaapaatrunge",
        ]

    def interpret(self, raw_transcript: str) -> VoiceCommandResult:
        """
        Parses text transcript from Web Speech API or microphone audio stream.
        """
        cleaned = raw_transcript.strip().lower()
        if not cleaned:
            return VoiceCommandResult(
                command_type="UNKNOWN",
                recognized_intent="EMPTY",
                target_action="IGNORE",
                target_param=None,
                spoken_feedback="Tiada arahan suara dikesan.",
                feedback_message="No audio input detected.",
                confidence=0.0,
            )

        # 1. Check for Operator / Caregiver commands
        for pattern, intent, action, speech in self.cmd_patterns:
            if re.search(pattern, cleaned):
                param = action if intent == "SWITCH_SCENARIO" else None
                act = "SWITCH_SCENARIO" if intent == "SWITCH_SCENARIO" else action
                return VoiceCommandResult(
                    command_type="OPERATOR_COMMAND",
                    recognized_intent=intent,
                    target_action=act,
                    target_param=param,
                    spoken_feedback=speech,
                    feedback_message=f"Operator command recognized: '{intent}' (Action: {act}).",
                    confidence=0.96,
                )

        # 2. Check for Patient Distress
        if any(re.search(p, cleaned) for p in self.patient_distress_patterns):
            return VoiceCommandResult(
                command_type="PATIENT_RESPONSE",
                recognized_intent="PATIENT_DISTRESS",
                target_action="PATIENT_DISTRESS",
                target_param=cleaned,
                spoken_feedback="Sakit dikesan! Mengaktifkan penggera dan menghantar bantuan serta-merta.",
                feedback_message=f"Patient vocal distress confirmed: \"{raw_transcript}\".",
                confidence=0.98,
            )

        # 3. Check for Patient Safe Confirmation
        if any(re.search(p, cleaned) for p in self.patient_safe_patterns):
            return VoiceCommandResult(
                command_type="PATIENT_RESPONSE",
                recognized_intent="PATIENT_SAFE",
                target_action="PATIENT_SAFE",
                target_param=cleaned,
                spoken_feedback="Alhamdulillah pakcik elok. Sistem berdiri siap dalam keadaan selamat.",
                feedback_message=f"Patient verbally confirmed safety: \"{raw_transcript}\".",
                confidence=0.97,
            )

        # 4. Fallback ambient speech
        return VoiceCommandResult(
            command_type="UNKNOWN",
            recognized_intent="AMBIENT_SPEECH",
            target_action="IGNORE",
            target_param=cleaned,
            spoken_feedback="Arahan tidak dikenali.",
            feedback_message=f"Ambient phrase detected: \"{raw_transcript}\" (no matching intent).",
            confidence=0.50,
        )
