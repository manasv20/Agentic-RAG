# LinkedIn Video Script — Screen Recording Analysis & Clip Map (90 sec)

## Your 3 Latest Screen Recordings (Desktop)

| Order | File | Size | Recorded (approx) |
|-------|------|------|-------------------|
| **1 (newest)** | `Screen Recording 2026-02-06 at 10.29.01 PM.mov` | ~57 MB | Feb 6, 2026, 10:29 PM |
| **2** | `Screen Recording 2026-02-06 at 10.22.16 PM.mov` | ~66 MB | Feb 6, 2026, 10:22 PM |
| **3 (oldest)** | `Screen Recording 2026-02-06 at 10.21.35 PM.mov` | ~13 MB | Feb 6, 2026, 10:21 PM |

*Note: I can’t view video content—this map is based on your script and file order. Watch each clip and tick the “Use for” row once you confirm what’s on screen.*

---

## Script → Clip Mapping (90 seconds total)

| Time | Section | Visual focus (from your script) | Audio / caption | Suggested source clip | Use for (✓ after you check) |
|------|---------|----------------------------------|-----------------|------------------------|-----------------------------|
| **0:00–0:05** | Hook | Video Processing: “Agentic brain: frame sampling” log | *“Standard RAG is static. Agentic RAG thinks for itself.”* | Clip 1 or 2 (whichever shows **Video Processing** + frame sampling) | _____ |
| **0:05–0:20** | Setup | Document Processing: uploading “LLM_Handbook” | *“Upload a complex doc, and let the agent decide the ingestion strategy.”* | Clip 3 or 2 (whichever shows **Document** upload + LLM_Handbook) | _____ |
| **0:20–0:45** | The “Brain” | Chunking: agent choosing e.g. 20/1630 pages to sample | *“Watch the agent reason: it samples the doc, identifies it as technical, and picks the best chunking style.”* | Clip 3 or 2 (whichever shows **Agent brain** / pipeline log + page sample count) | _____ |
| **0:45–1:15** | Result | Chat: question on author’s take on LLMs → “Agent Answer” | *“Clean retrieval from 1,900+ chunks. No hallucinations, just grounded context.”* | Clip 3 or 1 (whichever shows **Chat** + Q&A) | _____ |
| **1:15–1:30** | CTA | Navigation / Debug Panel / ChromaDB status | *“Built with Ollama, ChromaDB, and a custom agentic pipeline. Check out the repo below!”* | Clip 2 or 1 (whichever shows **sidebar** / Debug / ChromaDB) | _____ |

---

## How to use this

1. **Watch Clip 3** (10.21.35) — ~13 MB, shortest. Note: Document upload? Chunking/Agent brain? Chat?
2. **Watch Clip 2** (10.22.16) — ~66 MB. Note: Video Processing? Document? Sidebar/Debug?
3. **Watch Clip 1** (10.29.01) — ~57 MB. Note: Chat? Video? Sidebar?
4. **Fill the “Use for” column** above with the clip number (1, 2, or 3) for each section.
5. **Edit** in order: Hook (5s) → Setup (15s) → Brain (25s) → Result (30s) → CTA (15s) = 90s total.

---

## Quick reference — what to look for in each clip

- **Hook:** “Agent is thinking… (choosing frame sampling)” or “Agent brain: frame sampling” on **Video Processing** page.
- **Setup:** **Document Processing** — file uploader with “LLM_Handbook” (or similar) selected.
- **Brain:** Pipeline log or “Agent brain” expander showing “Agent chose to sample **N** pages (of P)” and chunking style.
- **Result:** **Chat** page with a question and an answer (ideally mentioning chunks or context).
- **CTA:** Sidebar with “Debug Panel” or “Clear ChromaDB” / collection list.

Repo: **github.com/manasv20/Agentic-RAG**

---

## Split & merge (your chosen segments)

Segments combined in order:

| Source | In point | Out point | Duration |
|--------|----------|-----------|----------|
| Video 1 (10.21.35) | 0:00 | 0:25 | 25 s |
| Video 2 (10.22.16) | 0:00 | 0:57 | 57 s |
| Video 2 (10.22.16) | 4:22 | 4:52 | 30 s |
| Video 3 (10.29.01) | 0:00 | 0:38 | 38 s |
| **Total** | | | **2 min 30 s** |

**Run the merge script** (from Terminal, with the 3 screen recordings on your Desktop):

```bash
cd "/Users/manasverma/Desktop/RAG Workshop/workshop demo"
./merge_video_segments.sh
```

Or pass your Desktop path:

```bash
./merge_video_segments.sh "/Users/manasverma/Desktop"
```

Output: **`LinkedIn_Agentic_RAG_Combined.mp4`** on your Desktop. Temp segments are in `Desktop/VideoMergeTemp` (you can delete that folder after).
