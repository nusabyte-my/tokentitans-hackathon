"""
GuardianBox - Malaysian Emergency Response Services (MERS 999) Engine
Official Malaysian Emergency Automation for Critical Geriatric Falls:
- Primary Hotline: MERS 999 (Kementerian Kesihatan Malaysia / PDRM / Bomba / APM)
- Secondary Welfare Hotline: Talian Kasih 15999 (KPWKM Senior Welfare)
- Formats statutory MERS 999 Operator Verbal Script (Bahasa Melayu & English)
- Computes Emergency Triage Category: KOD MERAH (Red Zone - Life Threatening)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


@dataclass
class MalaysianEmergencyContact:
    service_name: str
    hotline: str
    agency: str
    description: str
    priority: int
    is_toll_free: bool = True
    deep_link: str = ""


# Official Malaysian Emergency Services Directory
MALAYSIA_HOTLINES: Dict[str, MalaysianEmergencyContact] = {
    "MERS_999": MalaysianEmergencyContact(
        service_name="MERS 999 (Malaysian Emergency Response Services)",
        hotline="999",
        agency="KKM (Kementerian Kesihatan Malaysia) / PDRM / Bomba / APM",
        description="National Unified Emergency Number for Ambulance, Police, and Rescue.",
        priority=1,
        is_toll_free=True,
        deep_link="tel:999",
    ),
    "TALIAN_KASIH_15999": MalaysianEmergencyContact(
        service_name="Talian Kasih KPWKM",
        hotline="15999",
        agency="Kementerian Pembangunan Wanita, Keluarga dan Masyarakat",
        description="National Senior Citizen & Welfare Emergency Helpline (24/7).",
        priority=2,
        is_toll_free=True,
        deep_link="tel:15999",
    ),
    "TALIAN_KASIH_WHATSAPP": MalaysianEmergencyContact(
        service_name="Talian Kasih WhatsApp SOS",
        hotline="+60192615999",
        agency="KPWKM Welfare Dispatch",
        description="Direct WhatsApp intake for senior citizen welfare & crisis.",
        priority=3,
        is_toll_free=False,
        deep_link="https://wa.me/60192615999?text=Kecemasan%20Warga%20Emas%20GuardianBox",
    ),
    "ST_JOHN_AMBULANCE": MalaysianEmergencyContact(
        service_name="St. John Ambulance Malaysia (SJAM)",
        hotline="+60392851576",
        agency="Non-Governmental Emergency Medical Service",
        description="Private/NGO Emergency Ambulance Backup.",
        priority=4,
        is_toll_free=False,
        deep_link="tel:+60392851576",
    ),
}


@dataclass
class MERS999IncidentDossier:
    triage_code: str  # "KOD MERAH (Red Zone)" | "KOD KUNING (Yellow Zone)"
    emergency_service_needed: str  # "AMBULANS KECEMASAN (KKM)"
    patient_name: str
    patient_age: int
    home_address: str
    gps_coordinates: str
    incident_timestamp: str
    time_on_floor_seconds: float
    patient_responsiveness: str
    suspected_injuries: str
    mers999_verbal_script_bm: str
    mers999_verbal_script_en: str
    primary_hotline: str = "999"
    secondary_hotline: str = "15999"


class MalaysianEmergencyEngine:
    def __init__(self):
        self.default_address = os.getenv(
            "PATIENT_HOME_ADDRESS",
            "No. 42, Jalan Bukit Antarabangsa, 68000 Ampang, Selangor, Malaysia",
        )
        self.default_gps = os.getenv("PATIENT_GPS_COORDINATES", "3.1754° N, 101.7582° E")
        self.patient_name = os.getenv("PATIENT_FULL_NAME", "Haji Ahmad bin Sulaiman")
        self.patient_age = int(os.getenv("PATIENT_AGE", "76"))
        self.neighbor_phone = os.getenv("NEIGHBOR_PHONE", "+6012-345-6789 (Pak Ali)")

    def generate_dossier(
        self,
        time_on_floor_s: float,
        omni_reasoning: str,
        nano_audio_status: str,
        camera_location: str = "Living Room (Kamera Utama)",
    ) -> MERS999IncidentDossier:
        """
        Constructs an official, structured Malaysian 999 Incident Dispatch Dossier.
        Follows standard KKM/MERS emergency intake protocol.
        """
        now_str = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S MYT")
        is_unresponsive = (nano_audio_status == "UNRESPONSIVE")

        triage_code = "KOD MERAH (ZON KRITIKAL / RED ZONE)" if is_unresponsive else "KOD KUNING (ZON SEGERA)"

        responsiveness = (
            "TIDAK SEDARKAN DIRI / UNRESPONSIVE (Tiada tindak balas selepas semakan suara 10 saat)"
            if is_unresponsive
            else "SEPARUH SEDAR / MENGADUH KESAKITAN (Vocal distress detected)"
        )

        suspected = (
            "Kecederaan kepala terhimpit lantai, trauma femur/pinggul, hipotermia lantai sejuk."
            if is_unresponsive
            else "Trauma sendi pinggul dan kesukaran bernafas."
        )

        # Standard verbal script for caller to read to MERS 999 operator
        script_bm = (
            f"\"Kecemasan MERS 999! Sila hantar Ambulans KKM ke alamat: {self.default_address}. "
            f"Koordinat GPS: {self.default_gps}. "
            f"Mangsa: {self.patient_name}, umur {self.patient_age} tahun. "
            f"Keadaan: Rebah di lantai {camera_location} selama {time_on_floor_s:.0f} saat, "
            f"{responsiveness}. Disahkan oleh sistem AI peranti GuardianBox. Mohon bantuan segera!\""
        )

        script_en = (
            f"\"MERS 999 Emergency! Please dispatch a KKM Ambulance to: {self.default_address}. "
            f"GPS Coordinates: {self.default_gps}. "
            f"Patient: {self.patient_name}, age {self.patient_age}. "
            f"Condition: Fallen on floor in {camera_location} for {time_on_floor_s:.0f} seconds, "
            f"{responsiveness}. Automated tele-triage verified via GuardianBox on DGX Spark. Immediate dispatch required!\""
        )

        return MERS999IncidentDossier(
            triage_code=triage_code,
            emergency_service_needed="AMBULANS KECEMASAN (KKM HOSPITAL AMPANG)",
            patient_name=self.patient_name,
            patient_age=self.patient_age,
            home_address=self.default_address,
            gps_coordinates=self.default_gps,
            incident_timestamp=now_str,
            time_on_floor_seconds=round(time_on_floor_s, 1),
            patient_responsiveness=responsiveness,
            suspected_injuries=suspected,
            mers999_verbal_script_bm=script_bm,
            mers999_verbal_script_en=script_en,
            primary_hotline="999",
            secondary_hotline="15999",
        )

    def get_hotlines_list(self) -> List[Dict[str, Any]]:
        """Returns JSON list of Malaysian emergency hotlines."""
        return [
            {
                "id": k,
                "name": v.service_name,
                "hotline": v.hotline,
                "agency": v.agency,
                "description": v.description,
                "priority": v.priority,
                "deep_link": v.deep_link,
            }
            for k, v in sorted(MALAYSIA_HOTLINES.items(), key=lambda x: x[1].priority)
        ]
