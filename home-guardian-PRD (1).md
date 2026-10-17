# Home Guardian — PRD (v2)

**Event:** ACE 2026 Mini Hackathon · **Build time:** 3 hours · **Hardware:** Gigabyte AI TOP ATOM (one per team)
**Industry:** Healthcare / elder care
**One-liner:** A local AI that watches over elderly people living alone, detects falls with computer vision, confirms them with a vision-reasoning model, lets an agent decide what to do, and alerts family with text only. The video never leaves the house.

---

## 0. Hackathon brief (transcribed from the briefing slides)

### 0.1 The objective — "Local AI, Real Problems"
- Build a working AI application that solves a real-world problem and shows what local, on-premises AI can do.
- Show what Malaysian builders can achieve with private AI infrastructure.
- Go beyond simple LLM demos to agents, multimodal AI and real-world workflows.
- Explore any of: LLMs, VLMs, RAG, coding agents, speech-to-text, computer vision, automation, multi-model workflows.

### 0.2 Project requirements
1. Identify a real-world problem
2. Build a working AI application or prototype
3. Show meaningful AI beyond a basic chatbot
4. Run local inference on your AI TOP ATOM
5. Explain the practical value of running it locally
6. Demo the completed workflow

### 0.3 "Not that · No slop"
- A basic chatbot with nothing behind it
- A wrapper around a cloud LLM API
- Receipt scanners and calorie counters
- Anything that would run the same without the ATOM

### 0.4 Judging criteria (scored out of 100, same criteria for every team)
| Weight | Criterion | What judges look for |
|---|---|---|
| 25% | AI capability | Meaningful AI beyond a basic chatbot: agents, multimodal, RAG or multi-model workflows |
| 25% | Local AI on the ATOM | Inference runs on the ATOM, and you explain the practical value of running it locally |
| 20% | Impact & feasibility | A real-world problem, with relevance, sustainability and potential adoption |
| 15% | Technical implementation | A working application that demonstrates the completed workflow |
| 15% | Pitch & demo | Clarity of the 5-minute pitch, teamwork and GitHub documentation |

### 0.5 Suggested industries (open-ended — any real problem is allowed)
- **Healthcare** — patient data that should never leave the building
- **Insurance** — claims, documents and fraud checks, processed on-prem
- **Banking** — compliance, KYC and customer service with private data
- **Legal** — contracts and case files, reviewed without the cloud
- **Cybersecurity** — threat triage and log analysis that stays in-house
- **Developer productivity** — coding agents running on your own hardware

### 0.6 Hardware and provided stack — "One ATOM per team"
- Gigabyte AI TOP ATOM, NVIDIA DGX Spark platform
- GB10 Grace Blackwell Superchip
- 128 GB unified CPU + GPU memory
- 1 PFLOP FP4 AI compute
- Runs models up to 200B parameters locally
- Tip: the ATOM comes preconfigured with the baseline model and inference environment

| Layer | Provided / suggested |
|---|---|
| Baseline | NVIDIA Nemotron 3.5 Lightning |
| Advanced (optional) | NVIDIA NeMo Switchyard for multi-model routing |
| Serve | vLLM, Ollama or llama.cpp |
| Explore | Other open-source models and tools, e.g. Parakeet (speech), Cosmos Reason (vision) |
| References | build.nvidia.com/spark |

---

## 1. Compliance check

### 1.1 Requirements
| # | Requirement | Status | How Home Guardian meets it |
|---|---|---|---|
| R1 | Real-world problem | ✅ Follows | Unnoticed falls among elderly people living alone; existing pendants, watches and cloud CCTV each fail |
| R2 | Working app or prototype | ✅ Follows (must deliver) | Live camera → detection → confirmation → decision → alert, end to end |
| R3 | Meaningful AI beyond a chatbot | ✅ Follows | No chat interface at all. Pose CV + vision-reasoning model + LLM agent (+ speech) |
| R4 | Local inference on the ATOM | ✅ Follows | Every model runs on the ATOM. Only the Telegram text alert uses the network |
| R5 | Explain value of running locally | ✅ Follows (must say it in pitch) | Privacy of in-home video, works offline, always-on with no per-frame API cost, several models resident at once |
| R6 | Demo the completed workflow | ✅ Follows (must deliver) | 3-scene live demo: walk → prayer → collapse → alert |

### 1.2 "No slop" list
| Rule | Status | Notes |
|---|---|---|
| Basic chatbot | ✅ Not violated | No chatbot |
| Cloud LLM wrapper | ✅ Not violated | Zero cloud AI calls. Telegram only carries the final text |
| Receipt scanner / calorie counter | ✅ Not violated | — |
| Would run the same without the ATOM | ⚠️ Main risk — must be argued | A pose model alone would run on a laptop. The ATOM is needed for the **combination**: pose + vision-reasoning model + LLM agent + speech, all loaded at once, running 24/7. **Do not ship a pose-only demo**; the VLM and agent steps are mandatory |

### 1.3 Gaps found in v1 and fixed in v2
| Gap in v1 | Fix in v2 |
|---|---|
| Agent LLM was a "stretch" — without it the workflow is just "detector + classifier", weaker on the 25% AI criterion | Agent decision step with Nemotron is now **Must** |
| Used only third-party models; ignored the provided stack | Use the preconfigured **Nemotron 3.5 Lightning** as the agent, try **Cosmos Reason** as the VLM, **Parakeet** for speech |
| GitHub documentation listed late | README is a **Must**, with architecture, setup and "why local" sections — it's part of the 15% pitch score |
| Teamwork not mentioned | Pitch credits each member's part (see §11) |
| "Why local" only in slides | Must also appear on screen during the demo (stats panel) and in the README |

---

## 2. Problem
- Elderly parents often live alone while their children work in another city. A fall can go unnoticed for hours.
- **Panic pendants** only work if worn and pressed; a confused or unconscious person can't press them.
- **Smartwatches** need daily charging and remembering to wear.
- **Cloud CCTV** streams the home to external servers 24/7; many parents refuse to be watched.
- **Local nuance:** a naive fall detector flags kneeling in prayer (sujud) or sitting on the floor to rest, causing false alarms that make families ignore it.

## 3. Goals (mapped to judging)
| # | Goal | Criterion |
|---|---|---|
| G1 | Detect a person on the floor from a live camera feed | AI capability |
| G2 | Vision-reasoning model tells a fall apart from prayer or resting | AI capability |
| G3 | Agent decides alert vs stand down from all the evidence | AI capability |
| G4 | All models on the ATOM; only a text alert leaves the box | Local AI |
| G5 | Family gets a useful alert; nothing to wear or charge | Impact & feasibility |
| G6 | Working end-to-end workflow, 3 scenes | Technical implementation |
| G7 | 5-minute pitch + clean GitHub README | Pitch & demo |

## 4. Non-goals (out of scope for 3 hours)
- Bathroom monitoring (future audio-only mode)
- Night wandering and long-inactivity alerts
- Multi-camera / multi-room
- Fine-tuning
- Mobile app (Telegram is the family interface)

## 5. Users
- **Elderly person:** wears nothing, does nothing; only answers "Are you okay?"
- **Family member:** receives a short Telegram text describing what happened and when.

## 6. User flow
1. Camera watches the living area. Status `NORMAL`.
2. Pose model sees the body low/horizontal for N seconds → `SUSPECT`.
3. Vision-reasoning model reviews 3–4 recent frames → `fall` / `prayer` / `resting` / `normal` + one-sentence reason.
4. *(If speech is ready)* System asks aloud "Are you okay?" and transcribes the reply.
5. **Agent (Nemotron)** receives the VLM verdict, the reply transcript (or "no response") and the time on floor, and returns `ALERT` or `STAND_DOWN` with a message.
6. On `ALERT`: Telegram text to family + red banner on screen. No image or video is sent.

## 7. Functional requirements
| ID | Requirement | Priority |
|---|---|---|
| F1 | Webcam frames at ≥10 FPS | Must |
| F2 | YOLO pose on every frame; skeleton overlay | Must |
| F3 | `is_on_floor()` using box aspect ratio, torso angle, hip height | Must |
| F4 | Debounce: condition must hold N seconds | Must |
| F5 | Send 3–4 sampled frames to the vision model; parse JSON reply | Must |
| F6 | Prayer / kneeling classified as normal | Must |
| F7 | **Agent decision with Nemotron** (JSON in → JSON out) | Must |
| F8 | On-screen status, VLM reason, agent decision, latencies | Must |
| F9 | On-screen red alert banner | Must |
| F10 | Telegram text alert | Should |
| F11 | Stats panel: models loaded, memory used, per-model latency | Should |
| F12 | Voice check-in: TTS ask, Parakeet (or Whisper) listen, 10 s timeout | Could |
| F13 | Event log (timestamp, state, reason, decision) | Could |
| F14 | NeMo Switchyard routing between models | Won't (optional on brief; not worth the 3-hour risk) |

## 8. Non-functional requirements
- **Privacy:** no frame written to disk or sent off the machine; only alert text leaves.
- **Latency:** fall → alert in under 20 s (excluding the check-in wait).
- **Fail safe:** if the VLM or agent errors or returns invalid JSON, treat it as a fall and alert.
- **Offline:** everything except Telegram works with no internet.

## 9. Architecture
```
Webcam ─► YOLO pose (every frame) ─► is_on_floor? ─► debounce N s
                                                      │
                                                      ▼
                                   Vision model (Cosmos Reason / Qwen2.5-VL)
                                   fall / prayer / resting / normal + reason
                                                      │
                                     (optional) voice check-in: TTS ask → Parakeet STT
                                                      │
                                                      ▼
                                  Agent: Nemotron 3.5 Lightning (preconfigured)
                                  inputs: verdict, reply/no reply, seconds on floor
                                  output: ALERT or STAND_DOWN + message
                                                      │
                                     ┌────────────────┴──────────────┐
                                 STAND_DOWN                        ALERT
                                 log + CLEARED          Telegram text + screen banner
```

### 9.1 Model stack
| Stage | Primary choice | Fallback | Serve with |
|---|---|---|---|
| Pose / CV | YOLO pose `[n or s]` | — | ultralytics (NVIDIA PyTorch container on Arm64) |
| Vision reasoning | Cosmos Reason `[version/size]` | Qwen2.5-VL 7B | `[vLLM / Ollama]` |
| Agent LLM | Nemotron 3.5 Lightning (preinstalled) | `[any local LLM]` | whatever the ATOM is preconfigured with |
| STT (Could) | Parakeet `[model]` | Whisper `[size]` | `[NeMo / faster-whisper]` |
| TTS (Could) | Piper | pyttsx3 | CPU |

**First 15 minutes:** check what's already running on the ATOM (endpoint URL, port, model name for Nemotron) and whether Cosmos Reason has a ready playbook on build.nvidia.com/spark. If Cosmos Reason isn't up within 20 minutes, switch to Qwen2.5-VL via Ollama and move on.

### 9.2 Repo layout
```
home-guardian/
├── main.py      # loop: camera → pose → state machine → VLM → agent → alert, overlay
├── pose.py      # is_on_floor(kpts, box, frame_h) -> bool
├── vlm.py       # confirm(frames) -> {"state", "reason"}
├── agent.py     # decide(verdict, reply, seconds) -> {"action", "message"}
├── alert.py     # send_telegram(text), banner
├── checkin.py   # (Could) ask + listen
├── .env.example # VLM_URL=, AGENT_URL=, TELEGRAM_TOKEN=, TELEGRAM_CHAT_ID=
└── README.md
```

### 9.3 Prompts (v1)
**Vision model**
```
You watch over an elderly person at home.
Classify the scene as one of: fall, prayer, resting, normal.
Prayer = kneeling or forehead to floor, controlled posture.
Reply only JSON: {"state": "...", "reason": "<one sentence>"}
```
**Agent**
```
You are a home safety agent for an elderly person living alone.
Inputs: vision verdict, their spoken reply (or "no response"), seconds on the floor.
Decide ALERT or STAND_DOWN. When unsure, ALERT.
Write a short message for the family: what happened, where, when. No medical diagnosis.
Reply only JSON: {"action": "ALERT|STAND_DOWN", "message": "..."}
```

## 10. State machine
| State | Enter when | Exit to |
|---|---|---|
| `NORMAL` | Default | `SUSPECT` after N s on floor |
| `SUSPECT` | On-floor rule held N s | `CLEARED` or `FALL` after vision model |
| `CLEARED` | Verdict prayer/resting/normal | `NORMAL` when person stands |
| `FALL` | Verdict fall | agent → `ALERTED` or `CLEARED` |
| `ALERTED` | Agent said ALERT | `NORMAL` on manual reset (key `r`) |

Tunables: `FLOOR_SECONDS = __`, `FRAME_BUFFER = __`, `VLM_FRAMES = __`, `CHECKIN_TIMEOUT = 10`.

## 11. Build plan (3 hours)
| Time | Task | Done when |
|---|---|---|
| 0:00–0:20 | Find preconfigured Nemotron endpoint; container with CUDA on Arm64; webcam test; start vision model | Frame on screen; Nemotron and vision model each answer a test |
| 0:20–1:00 | Pose + `is_on_floor` + debounce | Lying 3 s → `SUSPECT` |
| 1:00–1:40 | Vision model confirm + prompt tuning | Kneel → `CLEARED`; collapse → `FALL` |
| 1:40–2:05 | Agent decision + banner + Telegram | Fall → agent `ALERT` → phone buzzes |
| 2:05–2:25 | Overlay + stats panel (models loaded, memory, latencies) | Readable from judges' seats |
| 2:25–2:45 | README (problem, architecture, setup, why local, team roles) | Pushed to GitHub |
| 2:45–3:00 | Backup demo video, rehearse 5-min pitch | Runs under 5 minutes |

**Cut order if late:** voice check-in → Telegram (keep banner) → stats panel. **Never cut** the vision model or the agent step — without them the project risks "would run the same without the ATOM".

**Suggested split (adjust to team size):** A = pose + state machine, B = vision model + agent, C = alert + overlay + README + pitch.

## 12. Pitch outline (5 minutes)
1. Problem (45 s) — falls, existing options fail, privacy refusal
2. Live demo (2 min) — walk, prayer, collapse → alert
3. How it works (45 s) — 3–4 models in one workflow, Nemotron agent
4. Why local (45 s) — video never leaves; offline; always-on with no API bill; all models resident in 128 GB
5. Impact + next + team (45 s) — families, care homes, clinics; bathroom audio mode; who built what

## 13. Acceptance criteria (mapped to judging)
- [ ] **AI capability:** pose, vision model and agent all run in the demo workflow
- [ ] **Local AI:** all inference on the ATOM; stats panel shows models loaded; `alert.py` sends text only
- [ ] **Impact:** pitch names the user, the alternative products and why they fail
- [ ] **Technical:** all three demo scenes give the expected result, 3 runs in a row
- [ ] **Pitch & demo:** under 5 minutes; README with architecture, setup, why local and team roles; backup video recorded

## 14. Risks and mitigations
| Risk | Mitigation |
|---|---|
| Judges think "a laptop could do this" | Show stats panel with every model resident and per-model latency; say the 24/7 + privacy + multi-model argument explicitly |
| Cosmos Reason setup eats time | 20-minute cap, then Qwen2.5-VL via Ollama |
| Can't find Nemotron endpoint | Ask organisers immediately; fallback to any local LLM in Ollama |
| PyTorch / ultralytics wheels fail on Arm64 | Use the NVIDIA PyTorch container from the start |
| Pose misses someone lying down | Use `s` model; camera high and angled down |
| Vision model calls prayer a fall | Describe prayer in the prompt; send more frames |
| Vision model too slow | Fewer frames, resize to 512 px |
| No webcam on the ATOM | Phone as IP camera (`cv2.VideoCapture("http://...")`) |
| Venue Wi-Fi blocks Telegram | Screen banner is the alert |
| Lighting breaks detection live | Play backup video |

## 15. Future work
- Audio-only bathroom mode (thud + calls for help)
- Night wandering and long-inactivity alerts
- NeMo Switchyard to route between models as more are added
- Multi-room: one box per home or care centre
- Weekly wellbeing summary for family (still text only)
