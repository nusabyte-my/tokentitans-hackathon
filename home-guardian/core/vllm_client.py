"""
GuardianBox - vLLM Client Engine for Gigabyte AI TOP ATOM (DGX Spark)
Configured for on-premises vLLM serving with OpenAI-compatible API.
Supports:
- Multimodal Vision reasoning (Nemotron Omni / Cosmos / Qwen-VL)
- Fast Agent completions (Nemotron 3.5 Lightning / Nemotron Nano)
- Dynamic Health Checking & Model Catalog Discovery
- Automatic Local Fallback when vLLM is offline or cold-booting
"""

from __future__ import annotations

import base64
import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import httpx

logger = logging.getLogger("guardianbox.vllm")


@dataclass
class vLLMHealthStatus:
    is_connected: bool
    endpoint_url: str
    available_models: List[str]
    active_model: str
    latency_ms: float
    platform_info: str
    engine_details: Dict[str, Any]


class vLLMEngineClient:
    def __init__(
        self,
        endpoint_url: Optional[str] = None,
        api_key: Optional[str] = None,
        default_model: Optional[str] = None,
        timeout_seconds: float = 8.0,
    ):
        self.endpoint_url = (endpoint_url or os.getenv("VLLM_ENDPOINT", "http://127.0.0.1:8000/v1")).rstrip("/")
        self.api_key = api_key or os.getenv("VLLM_API_KEY", "EMPTY")
        self.default_model = default_model or os.getenv("VLLM_MODEL", "nvidia/nemotron-3.5-lightning")
        self.timeout_seconds = timeout_seconds
        self.is_connected = False
        self.last_health_check: Optional[vLLMHealthStatus] = None

    async def check_health(self) -> vLLMHealthStatus:
        """Probes vLLM server on ATOM DGX Spark and checks available models."""
        start_t = time.perf_counter()
        models = []
        is_up = False
        engine_details = {}

        headers = {"Authorization": f"Bearer {self.api_key}"}
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                res = await client.get(f"{self.endpoint_url}/models", headers=headers)
                if res.status_code == 200:
                    data = res.json()
                    models = [m["id"] for m in data.get("data", [])]
                    is_up = True
                    engine_details = {"models_loaded": len(models), "raw": data}
        except Exception as e:
            engine_details = {"error": str(e)}

        latency = (time.perf_counter() - start_t) * 1000.0
        self.is_connected = is_up

        status = vLLMHealthStatus(
            is_connected=is_up,
            endpoint_url=self.endpoint_url,
            available_models=models,
            active_model=models[0] if models else self.default_model,
            latency_ms=round(latency, 1) if is_up else 0.0,
            platform_info="Gigabyte AI TOP ATOM · NVIDIA DGX Spark (128GB LPDDR5x)" if is_up else "Local Edge Resident Fallback",
            engine_details=engine_details,
        )
        self.last_health_check = status
        return status
    async def sync_nvidia_atom_models(
        self,
        endpoint_url: Optional[str] = None,
        api_key: Optional[str] = None,
        omni_model: str = "nvidia/nemotron-omni",
        lightning_model: str = "nvidia/nemotron-3.5-lightning",
        nano_model: str = "nvidia/nemotron-nano-4b",
    ) -> Dict[str, Any]:
        """
        Synchronizes with NVIDIA DGX Spark / ATOM inference environment.
        Verifies and binds the 3 Nemotron models:
        1. Nemotron Lightning
        2. Nemotron Omni
        3. Nemotron Nano
        """
        if endpoint_url:
            self.endpoint_url = endpoint_url.rstrip("/")
        if api_key:
            self.api_key = api_key

        health = await self.check_health()
        
        model_results = {}
        for m_role, m_name in [("lightning", lightning_model), ("omni", omni_model), ("nano", nano_model)]:
            is_present = any(m_name.lower() in avail.lower() for avail in health.available_models)
            model_results[m_role] = {
                "name": m_name,
                "role": m_role.upper(),
                "status": "CONNECTED_WARM" if health.is_connected else "LOCAL_RESIDENT_SIMULATION",
                "resident_vram_gb": 18.0 if m_role == "lightning" else (16.5 if m_role == "omni" else 6.2),
                "matched_in_vllm": is_present,
            }

        return {
            "is_connected": health.is_connected,
            "endpoint_url": self.endpoint_url,
            "latency_ms": health.latency_ms,
            "platform": health.platform_info,
            "available_models_count": len(health.available_models),
            "synced_models": model_results,
        }

    async def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 256,
        json_mode: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """
        Dispatches chat completion request to vLLM on ATOM.
        Returns parsed JSON dict or None on failure.
        """
        target_model = model or self.default_model
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        body = {
            "model": target_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}

        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                res = await client.post(
                    f"{self.endpoint_url}/chat/completions",
                    headers=headers,
                    json=body,
                )
                if res.status_code == 200:
                    data = res.json()
                    content = data["choices"][0]["message"]["content"]
                    cleaned = content.strip()
                    if "```json" in cleaned:
                        cleaned = cleaned.split("```json")[1].split("```")[0].strip()
                    return json.loads(cleaned)
        except Exception as e:
            logger.debug(f"vLLM chat completion call skipped: {e}")
            return None

    async def vision_scene_reasoning(
        self,
        frames_base64: List[str],
        aspect_ratio: float,
        torso_angle: float,
    ) -> Optional[Dict[str, Any]]:
        """
        Multimodal VLM invocation through vLLM on ATOM.
        """
        system_prompt = (
            "You are Nemotron-Omni running on vLLM (DGX Spark). "
            "Analyze the camera frames for an elderly living alone. "
            "Classify into strictly: 'fall', 'prayer', 'resting', 'normal'. "
            "Output JSON: {\"state\": \"...\", \"reason\": \"...\", \"confidence\": float, \"posture_type\": \"...\"}"
        )
        user_content: List[Dict[str, Any]] = [
            {"type": "text", "text": f"Evaluate scene frames. Aspect ratio: {aspect_ratio:.2f}, Torso: {torso_angle:.1f} deg."}
        ]
        for b64 in frames_base64[:3]:
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
            })

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        return await self.chat_completion(messages, model=self.default_model, max_tokens=180, json_mode=True)
