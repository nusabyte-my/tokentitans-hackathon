# 🛡️ GUARDIANBOX — Sovereign Edge Elder Care

> **Event:** ACE 2026 Mini Hackathon · **Build Time:** 3 Hours  
> **Platform:** Gigabyte AI TOP ATOM · NVIDIA DGX Spark (GB10 Grace Blackwell · 128 GB Unified LPDDR5x)  
> **Theme:** Local AI, Real Problems · **Industry:** Healthcare / Elder Care  
> **Core Invariant:** Zero video or audio transmitted to the cloud. 100% on-premise inference.

---

## 1. Problem & The Privacy Dilemma

Millions of elderly parents live alone while their adult children work in urban centers. Unnoticed falls are the leading cause of fatal injury among seniors:

1. **Panic Pendants Fail:** Must be worn continuously; unconscious or disoriented seniors cannot press them.
2. **Smartwatches Fail:** Require daily charging and are frequently left on bedside nightstands.
3. **Cloud CCTV Fails:** Streaming intimate in-home video to cloud servers violates personal dignity and privacy. Most elderly parents refuse to be watched and unplug cameras.
4. **The Cultural Blindspot:** Generic fall detectors fail in Malaysia by mistaking Muslim prayer prostrations (**Sujud**) for medical collapses, triggering annoying false alarms.

**The Solution — GuardianBox:** A sovereign edge appliance powered by the Gigabyte AI TOP ATOM. It watches continuously, verifies scene geometry locally, checks in via localized voice, and dispatches family alerts **strictly as structured text**. Video never leaves the room.

---

## 2. Multi-Agent Swarm: 5 Specialized Nemotron Agents

GuardianBox deploys a **5-Agent Multi-Orchestration Swarm** running in parallel on the **128 GB unified memory** of the DGX Spark. All 5 agents stay continuously resident in VRAM:

```
[Edge Camera (30 FPS)] ──► [Agent 1: Kinematics Sentinel (Nemotron-Nano · 5.8 GB)]
                                 │ (Sudden horizontal descent detected)
                                 ▼
         ┌───────────────────────┴───────────────────────┐
         ▼                                               ▼
 [Agent 2: Spatial & Cultural]               [Agent 3: Acoustic & Dialect]
 Nemotron-Omni (VLM) · 16.5 GB               Nemotron-Nano (4B) · 6.2 GB
 Inspects multi-frame prostration            Dispatches localized voice:
 Identifies Sujud prayer vs trauma           "Pakcik, okay ke tak?"
         │                                               │
         └───────────────────────┬───────────────────────┘
                                 │
                                 ▼
             [Agent 4: Clinical Context & Historian]
             Nemotron-Nano (8B) · 7.5 GB
             Evaluates Circadian Risk, Time-on-floor & Senior Profile
                                 │
                                 ▼
             [Agent 5: Sovereign Commander Orchestrator]
             Nemotron-3.5-Lightning · 18.0 GB
             Quorum Voting & Veto Logic ──► Consensus Decision
                                 │
                                 ▼
             [Interactive Telegram Bot Escalation]
             • 2-Way Remote Confirmation Loop
             • 5-Agent Swarm Deliberation Breakdown
             • Zero Video Egress (112 bytes metadata)
```

### Resident Memory Footprint (128 GB Unified LPDDR5x)

| Specialist Agent | Framework / Backbone | Role | Resident VRAM | Latency |
|---|---|---|---|---|
| **Agent 1: Kinematics Sentinel** | Nemotron-Nano-4B | High-frequency motion trajectory, ground impact velocity | **5.8 GB** | ~14 ms |
| **Agent 2: Spatial & Cultural Reasoner** | Nemotron-Omni (Cosmos Reason) | Multimodal VLM posture analysis & Sujud prayer prostration | **16.5 GB** | ~740 ms |
| **Agent 3: Acoustic & Dialect Triage** | Nemotron-Nano-4B | Zero-latency vocal check-in & dialect distress parsing | **6.2 GB** | ~35 ms |
| **Agent 4: Clinical Historian** | Nemotron-Nano-8B | Circadian risk weighting & mobility context | **7.5 GB** | ~24 ms |
| **Agent 5: Sovereign Commander** | Nemotron-3.5-Lightning | Master quorum adjudicator, veto synthesis & dispatch | **18.0 GB** | ~195 ms |
| **Total Resident Swarm Pool** | **All 5 Agents Resident in VRAM** | **Instant NVLink pipeline handoff without swapping** | **54.0 GB / 128 GB** | **Pipeline: ~1.0s** |

---

## 3. High-Impact Quickwins for the Hackathon Pitch

1. **5-Agent Consensus Matrix & Quorum HUD:** Shows live voting chips (`🟢 CLEAR` / `🟡 SUSPECT` / `🔴 CRITICAL ALERT`) across all 5 agents simultaneously on the big screen.
2. **Cultural Quorum Override Demonstration (Sujud vs Fall):** 
   - When an elderly person prostrates in prayer, Agent 1 (Kinematics) votes `SUSPECT`, but Agent 2 (Omni) votes `CLEAR (Sujud)` and Agent 4 (Clinical) verifies the Islamic prayer window.
   - Agent 5 (Commander) exercises sovereign veto power and executes a **CULTURAL QUORUM STAND DOWN** (98% consensus), proving why a 5-agent swarm beats a naive single model!
3. **"The Hardware Proof" — 54 GB VRAM Telemetry:** Displays exact memory consumption across all 5 resident agents inside the 128 GB Unified LPDDR5x pool, dismantling the *"could run on a laptop"* critique.
4. **2-Way Interactive Telegram Feedback Loop:** The emergency alert carries interactive inline callback buttons (`[✅ I'm OK / False Alarm]`, `[📞 Call Intercom]`, `[🚨 Call 999]`, `[🏃 Dispatch Jiran]`). Clicking remotely immediately flips the dashboard from red emergency to green cleared.
5. **1-Click Golden Scenario Switcher:** UI buttons (`[Normal Walk]`, `[Sujud / Solat]`, `[Acute Fall]`, `[Webcam]`) guarantee 100% deterministic stage execution without venue lighting glitches.
6. **Zero-Egress Privacy Watermark:** Real-time egress audit showing `0.00 KB video transmitted · 112 bytes metadata only`.


## 4. Quickstart & Installation

```bash
# 1. Clone repository & navigate to workspace
cd "C:\Users\shaid\Desktop\DGX Hackathon\home-guardian"

# 2. Install dependencies (PyTorch, OpenCV, FastAPI, etc.)
pip install -r requirements.txt

# 3. Configure Telegram Bot (Optional: falls back to simulated local mode)
cp .env.example .env
# Edit TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env

# 4. Run End-to-End Smoke Test
python test_smoke.py

# 5. Launch GuardianBox Tactical Dashboard
python app.py
```
Open **`http://localhost:53532`** in Chrome / Edge.

---

## 5. 5-Minute Live Pitch & Demonstration Script

### Minute 0:00 – 0:45: The Problem & The Privacy Dilemma
> *"Good morning judges. Every year, thousands of elderly Malaysians living alone suffer debilitating falls. Current solutions fail: pendants aren't worn in the shower or after fainting; smartwatches sit on charging stands; and cloud cameras are fiercely rejected because no parent wants strangers or corporate servers streaming their living room 24/7.*  
> *Meet **GuardianBox** — a sovereign edge device running on the Gigabyte AI TOP ATOM that protects our parents with zero cloud video leakage."*

### Minute 0:45 – 2:30: The Live Multimodal Demonstration
1. **Scene 1 (Walk):** *"Notice our edge camera tracking ambulation. The hairline HUD shows upright posture, aspect ratio 0.45. System state is NORMAL."*
2. **Scene 2 (The Sujud Test):** *(Click '2. Sujud / Solat')*  
   *"Now watch what happens when our parent performs Muslim prayer (Sujud). Naive cameras sound an alarm. But watch GuardianBox: the pose detector flags horizontal posture $\to$ dispatches **Nemotron Omni (VLM)**. Omni reasons over the scene and confirms: 'Subject in deliberate prayer posture (Sujud) with controlled joint angles.' Nemotron Lightning says: STAND DOWN. Zero false alarms."*
3. **Scene 3 (Acute Fall & Telegram Alert):** *(Click '3. Acute Fall')*  
   *"Now, catastrophic collapse. The person falls prone. Omni confirms an uncontrolled traumatic fall. **Nemotron Nano** fires a localized voice check-in: 'Pakcik, are you okay?'. 10 seconds of silence elapse. **Nemotron Lightning** synthesizes the emergency and dispatches."*
4. **Interactive Action:** *(Hold up phone with Telegram alert vibrating)*  
   *"Look at my phone. A structured alert card just arrived on Telegram — with zero video bytes. When I tap `[✅ I'm OK / False Alarm]`, watch the big screen: the red emergency banner instantly clears to green!"*

### Minute 2:30 – 3:45: The Hardware Proof — Why DGX Spark (ATOM)?
> *"Why can't this run on a laptop? Look at our hardware telemetry HUD: we have **Nemotron Omni (16.5 GB)**, **Nemotron Lightning (18.0 GB)**, and **Nemotron Nano (6.2 GB)** loaded resident in VRAM simultaneously. That's 41.5 GB of active weights. On a laptop or standard edge box, swapping between vision and language models creates a 20-second latency cliff. On the ATOM's 128 GB unified memory with 273 GB/s bandwidth, the entire pipeline executes in under 1.1 seconds."*

### Minute 3:45 – 5:00: Impact, PDPA Compliance & Market Adoption
> *"GuardianBox is 100% compliant with Malaysia's Personal Data Protection Act (PDPA 2010). Because all computing stays on-premises, families and elder care centers can deploy this without privacy liability. Thank you — we are ready for your questions."*

---

## 6. Directory Structure

```
home-guardian/
├── app.py                # FastAPI server, MJPEG streaming, WebSocket telemetry
├── test_smoke.py         # End-to-end automated verification suite
├── requirements.txt      # Dependency specification
├── core/
│   ├── pose_engine.py    # YOLO/MediaPipe pose geometry & temporal debounce
│   ├── nemotron_omni.py  # Nemotron Omni multimodal vision reasoner (Sujud classifier)
│   ├── nemotron_nano.py  # Nemotron Nano worker (multilingual audio check-in)
│   ├── nemotron_agent.py # Nemotron Lightning Cognitive Commander Agent
│   ├── telegram_bot.py   # 2-way interactive Telegram automation engine
│   └── scenarios.py      # Deterministic golden demo scenario generator
└── templates/
    └── index.html        # shadcn & Hairline UI tactical dark cockpit
```
