# Getting Started — Agentic RAG Workshop

**Follow these steps in order.** Instructions are given for **Windows (PowerShell)** and **macOS (Terminal)**.

---

## What You Need Before Starting

- **Python 3.10 or 3.11** (3.12 OK; avoid 3.9 or older)
- **Git** (to clone the repo)
- **Windows:** PowerShell  
- **macOS:** Terminal (Homebrew recommended for installing ffmpeg)

---

## Step 1: Clone the Repository

**Windows (PowerShell):**
```powershell
git clone https://github.com/manasv20/Agentic-RAG.git
cd Agentic-RAG
```

**macOS (Terminal):**
```bash
git clone https://github.com/manasv20/Agentic-RAG.git
cd Agentic-RAG
```

*(If you already have the repo, skip to Step 2.)*

---

## Step 2: Create and Activate a Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
*If you see an execution policy error, run: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` once.*

**macOS (Terminal):**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

You should see `(.venv)` in your prompt. All following commands assume this environment is active.

---

## Step 3: Install Python Dependencies

**Windows:**
```powershell
pip install -r requirements.txt
```

**macOS:**
```bash
pip install -r requirements.txt
```

This installs Streamlit, ChromaDB, Ollama client, PyPDF2, and (for full features) audio/video libraries.  
**Minimal install (documents + chat only):**
```bash
pip install streamlit PyPDF2 chromadb==0.4.24 ollama "numpy>=1.22,<2" python-dotenv
```

---

## Step 4: Install FFmpeg (Required for Audio and Video)

**Windows:**
- **Option A (Chocolatey):** `choco install ffmpeg`
- **Option B (Scoop):** `scoop install ffmpeg`
- **Option C (Manual):** Download from https://ffmpeg.org/download.html → extract → add the `bin` folder to your system PATH

**macOS:**
```bash
brew install ffmpeg
```

**Verify:** Run `ffmpeg -version` in a new terminal; you should see version info.

---

## Step 5: Install and Start Ollama

1. **Download and install Ollama** from https://ollama.ai (Windows or macOS).
2. **Start Ollama** (it usually runs in the background after install).
3. **Pull at least one model** for embeddings and chat:

**Windows and macOS:**
```bash
ollama pull phi
ollama pull gemma3:4b
```

*Use `phi` for embeddings; use `gemma3:4b` or a tool-capable model (e.g. `qwen3:4b`) for chat.*

---

## Step 6: Verify Setup (Optional but Recommended)

**Windows:**
```powershell
python scripts/test_audio_setup.py
python scripts/check_chromadb.py
```

**macOS:**
```bash
python3 scripts/test_audio_setup.py
python3 scripts/check_chromadb.py
```

Fix any reported issues before running the app.

---

## Step 7: Run the Application

**Windows:**
```powershell
streamlit run localragdemo.py
```

**macOS:**
```bash
streamlit run localragdemo.py
```

Your browser should open to **http://localhost:8501**. If not, open that URL manually.

---

## What to Do in the App

1. **Document Processing** — Upload a PDF, choose or create a collection, process. The agent picks chunk size and style from a sample.
2. **Audio Processing** — Upload audio, transcribe, same agentic chunking; add to a collection.
3. **Video Processing** — Upload video; agent picks frame rate; frames and/or transcript go to a collection.
4. **Chat** — Select a collection (document, audio, or video), ask questions. The agent can search multiple times; the Evaluator checks that answers are grounded.

---

## Troubleshooting

| Issue | Windows | macOS |
|-------|---------|--------|
| `python` not found | Use `py -3` or install Python from python.org and check "Add to PATH" | Use `python3` |
| Venv activate fails | Run `Set-ExecutionPolicy RemoteSigned -Scope CurrentUser` | Ensure you're in the project folder and use `source .venv/bin/activate` |
| FFmpeg not found | Add FFmpeg `bin` to system PATH | `brew install ffmpeg` |
| Ollama connection error | Start the Ollama app from Start menu | Run `ollama serve` or open Ollama from Applications |
| ChromaDB errors | Ensure `chromadb==0.4.24` and `numpy>=1.22,<2` | Same |

For more detail, see **README.md**, **COMMANDS.md**, and **AUDIO_GUIDE.md**.
