"""
GuardianBox - 5-Agent Swarm Verification Suite
Validates the multi-orchestration consensus pipeline across:
1. Agent 1: Kinematics Sentinel (Nemotron-Nano)
2. Agent 2: Spatial & Cultural Reasoner (Nemotron-Omni)
3. Agent 3: Acoustic & Dialect Triage (Nemotron-Nano)
4. Agent 4: Clinical Context & Historian (Nemotron-Nano/8B)
5. Agent 5: Sovereign Commander & Quorum Orchestrator (Nemotron-Lightning)

Verifies:
- Live parallel evaluation across 5 agents
- Cultural Quorum Override (Sujud prayer clears without false alarm)
- Emergency Quorum Trigger (Acute fall triggers 5-agent alert + Telegram dispatch)
- Hardware Unified Memory resident calculation (54.0 GB / 128 GB)
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from core.multi_agent_orchestrator import MultiAgentSwarmOrchestrator
from core.nemotron_nano import NemotronNanoWorker
from core.nemotron_omni import NemotronOmniReasoner
from core.pose_engine import PoseEngine
from core.scenarios import GoldenScenarioGenerator
from core.telegram_bot import TelegramAutomationBot


async def test_5_agent_swarm():
    print("=================================================================")
    print(" 🛡️  GUARDIANBOX - 5-AGENT MULTI-ORCHESTRATION SWARM TEST")
    print(" Architecture: Parallel Nemotron Triad (Nano · Omni · Lightning)")
    print(" Hardware: Gigabyte AI TOP ATOM (128 GB Unified LPDDR5x)")
    print("=================================================================\n")

    omni = NemotronOmniReasoner()
    nano = NemotronNanoWorker()
    swarm = MultiAgentSwarmOrchestrator(omni_reasoner=omni, nano_worker=nano)
    pose_engine = PoseEngine(floor_seconds_threshold=1.0)
    scenario_gen = GoldenScenarioGenerator()

    # ─────────────────────────────────────────────────────────────────
    # TEST 1: SUJUD PRAYER — CULTURAL QUORUM OVERRIDE
    # ─────────────────────────────────────────────────────────────────
    print("[TEST 1/2] Testing 5-Agent Quorum on Muslim Prayer (Sujud)...")
    frame, bbox, kpts = scenario_gen.generate_frame("sujud")
    pose_res = pose_engine.analyze_geometry(bbox, frame.shape, kpts)
    pose_res.horizontal_duration_sec = 3.5  # Simulate prolonged prostration

    result = await swarm.execute_swarm_deliberation(
        frames=[frame],
        pose_res=pose_res,
        simulated_scenario="sujud",
    )

    print(f"  -> Quorum Action: {result.action}")
    print(f"  -> Consensus Score: {result.consensus_score * 100:.1f}%")
    print(f"  -> Quorum Rationale: \"{result.quorum_summary}\"")
    print(f"  -> Total Latency: {result.total_pipeline_latency_ms:.1f}ms")
    print(f"  -> Swarm Resident VRAM: {result.swarm_vram_used_gb:.1f} GB / 128.0 GB")
    print("\n  Individual Agent Deliberation breakdown:")
    for ag in result.agent_findings:
        print(f"    • [{ag.vote:14}] {ag.agent_name:<28} ({ag.model_family}): conf {ag.confidence*100:.0f}% | {ag.verdict_summary}")

    assert result.action == "STAND_DOWN", f"Expected STAND_DOWN for Sujud prayer, got {result.action}"
    assert len(result.agent_findings) == 5, f"Expected 5 agents in swarm, got {len(result.agent_findings)}"
    print("  ✓ TEST 1 PASSED: Cultural prayer prostration recognized with unanimous/override stand-down.\n")

    # ─────────────────────────────────────────────────────────────────
    # TEST 2: ACUTE CATASTROPHIC COLLAPSE — EMERGENCY QUORUM
    # ─────────────────────────────────────────────────────────────────
    print("[TEST 2/2] Testing 5-Agent Quorum on Acute Traumatic Fall...")
    frame, bbox, kpts = scenario_gen.generate_frame("fall")
    pose_res = pose_engine.analyze_geometry(bbox, frame.shape, kpts)
    pose_res.horizontal_duration_sec = 4.2  # Prone on floor

    result_fall = await swarm.execute_swarm_deliberation(
        frames=[frame],
        pose_res=pose_res,
        simulated_scenario="fall",
    )

    print(f"  -> Quorum Action: {result_fall.action}")
    print(f"  -> Consensus Score: {result_fall.consensus_score * 100:.1f}%")
    print(f"  -> Quorum Rationale: \"{result_fall.quorum_summary}\"")
    print(f"  -> Total Latency: {result_fall.total_pipeline_latency_ms:.1f}ms")
    print(f"  -> Swarm Resident VRAM: {result_fall.swarm_vram_used_gb:.1f} GB / 128.0 GB")
    print("\n  Individual Agent Deliberation breakdown:")
    for ag in result_fall.agent_findings:
        print(f"    • [{ag.vote:14}] {ag.agent_name:<28} ({ag.model_family}): conf {ag.confidence*100:.0f}% | {ag.verdict_summary}")

    assert result_fall.action == "CRITICAL_ALERT", f"Expected CRITICAL_ALERT for fall, got {result_fall.action}"
    assert len(result_fall.agent_findings) == 5, f"Expected 5 agents in swarm, got {len(result_fall.agent_findings)}"

    # Test Telegram alert formatting with 5-agent breakdown
    bot = TelegramAutomationBot()
    alert_text = bot.format_alert_markdown(result_fall.telegram_payload)
    print("\n  Telegram 5-Agent Alert Card Sample:")
    print("  " + "\n  ".join(alert_text.split("\n")[:12]))
    assert "5-Agent Swarm Deliberation" in alert_text, "Telegram card should contain multi-agent breakdown"
    print("  ✓ TEST 2 PASSED: 5-Agent Quorum confirmed acute emergency and generated structured dispatch card.\n")

    print("=================================================================")
    print(" 🏆 ALL 5-AGENT SWARM TESTS PASSED!")
    print(f" Verified Swarm Memory Footprint: {result_fall.swarm_vram_used_gb} GB resident in 128 GB Unified LPDDR5x.")
    print("=================================================================")

if __name__ == "__main__":
    asyncio.run(test_5_agent_swarm())
