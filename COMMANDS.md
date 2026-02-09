# Quick Command Reference — Windows & macOS

Use this page for copy-paste commands. For a full step-by-step guide, see **[GETTING_STARTED.md](./GETTING_STARTED.md)**.

---

## What you need to do (summary)

| Step | Windows (PowerShell) | macOS (Terminal) |
|------|------------------------|------------------|
| 1. Clone | `git clone https://github.com/manasv20/Agentic-RAG.git` then `cd Agentic-RAG` | Same |
| 2. Venv | `python -m venv .venv` then `.\.venv\Scripts\Activate.ps1` | `python3 -m venv .venv` then `source .venv/bin/activate` |
| 3. Deps | `pip install -r requirements.txt` | Same |
| 4. FFmpeg | `choco install ffmpeg` or download and add to PATH | `brew install ffmpeg` |
| 5. Ollama | Install from ollama.ai, then `ollama pull phi` and `ollama pull gemma3:4b` | Same |
| 6. Run | `streamlit run localragdemo.py` | Same |

---

## Installation Commands

### Install Python Dependencies
```powershell
# All audio dependencies
pip install SpeechRecognition pydub

# Or use the requirements file
pip install -r requirements.txt
```

### Install FFmpeg (Choose One Method)

#### Windows - Chocolatey
```powershell
choco install ffmpeg
```

#### Windows - Scoop
```powershell
scoop install ffmpeg
```

#### Windows - Manual
1. Download from: https://ffmpeg.org/download.html
2. Extract to `C:\ffmpeg`
3. Add `C:\ffmpeg\bin` to PATH
4. Restart terminal

#### macOS
```bash
brew install ffmpeg
```

#### Linux (Ubuntu/Debian)
```bash
sudo apt-get update
sudo apt-get install ffmpeg
```

### Verify Installation
**Windows:**
```powershell
python scripts/test_audio_setup.py
# Or: .\install_audio_dependencies.ps1
```

**macOS / Linux:**
```bash
python3 scripts/test_audio_setup.py
# Or: ./install_audio_dependencies.sh
```

## Run Commands

### Start Application
**Windows:**
```powershell
# Activate virtual environment (if not already active)
.\.venv\Scripts\Activate.ps1

# Run Streamlit app
streamlit run localragdemo.py
```

**macOS / Linux:**
```bash
# Activate virtual environment (if not already active)
source .venv/bin/activate

# Run Streamlit app
streamlit run localragdemo.py
```

### Access Audio Processing
1. Open browser (usually auto-opens)
2. Click **"🎤 Audio Processing"** in sidebar
3. Upload audio file
4. Configure and transcribe

**Agent autonomy:** For documents the agent decides how many pages to sample; for audio/video the agent decides how much of the transcript (characters) to sample before choosing chunking. No fixed sample sizes—the agent has full control.

## Testing Commands

### Test Audio Setup
```powershell
python scripts/test_audio_setup.py
```

### Check ChromaDB
```powershell
python scripts/check_chromadb.py
```

### Verify Python Environment
**Windows:**
```powershell
python --version
pip list | Select-String "speech|pydub|chromadb|ollama"
ffmpeg -version
```

**macOS / Linux:**
```bash
python3 --version
pip list | grep -E "speech|pydub|chromadb|ollama"
ffmpeg -version
```

## Troubleshooting Commands

### Reinstall Audio Dependencies
```powershell
pip uninstall SpeechRecognition pydub -y
pip install SpeechRecognition pydub
```

### Clear ChromaDB (Careful!)
**Windows:**
```powershell
Copy-Item -Recurse chroma_db chroma_db_backup
Remove-Item -Recurse -Force chroma_db
```

**macOS / Linux:**
```bash
cp -R chroma_db chroma_db_backup
rm -rf chroma_db
```

### Check Logs
**Windows:**
```powershell
Get-Content (Get-ChildItem logs | Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName -Tail 50
```

**macOS / Linux:**
```bash
tail -50 $(ls -t logs/*.log 2>/dev/null | head -1)
```

### View Streamlit Cache
```powershell
# Clear Streamlit cache
streamlit cache clear
```

## Common Workflows

### First Time Setup
**Windows:**
```powershell
cd Agentic-RAG
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install SpeechRecognition pydub
choco install ffmpeg   # or scoop install ffmpeg
python scripts/test_audio_setup.py
streamlit run localragdemo.py
```

**macOS / Linux:**
```bash
cd Agentic-RAG
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install SpeechRecognition pydub
brew install ffmpeg
python3 scripts/test_audio_setup.py
streamlit run localragdemo.py
```

### Daily Usage
**Windows:**
```powershell
cd path\to\Agentic-RAG
.\.venv\Scripts\Activate.ps1

# 3. Run application
streamlit run localragdemo.py
```

**macOS / Linux:**
```bash
cd Agentic-RAG
source .venv/bin/activate
streamlit run localragdemo.py
# When done: deactivate
```

### Update Application
```powershell
# 1. Pull latest changes
git pull

# 2. Update dependencies
pip install -r requirements.txt --upgrade

# 3. Verify setup
python scripts/test_audio_setup.py

# 4. Run application
streamlit run localragdemo.py
```

## Environment Variables

### Set ChromaDB Path
**Windows:**
```powershell
$env:CHROMA_PATH = "D:\MyData\vector_db"
```

**macOS / Linux:**
```bash
export CHROMA_PATH="$HOME/data/vector_db"
```

### Set Default Collection
```powershell
$env:COLLECTION_NAME = "my_collection"
```

### Set Models
```powershell
$env:EMBEDDING_MODEL = "phi"
$env:LLM_MODEL = "gemma:2b"
```

## Quick Checks

### Is Python Working?
```powershell
python --version
# Expected: Python 3.10+ 
```

### Is Virtual Environment Active?
```powershell
$env:VIRTUAL_ENV
# Should show path to .venv
```

### Are Audio Libraries Installed?
```powershell
python -c "import speech_recognition; import pydub; print('OK')"
# Expected: OK
```

### Is FFmpeg Available?
```powershell
ffmpeg -version
# Should show version info
```

### Is ChromaDB Working?
```powershell
python -c "import chromadb; print('ChromaDB version:', chromadb.__version__)"
# Should show version
```

### Is Ollama Running?
```powershell
python -c "import ollama; print('Ollama OK')"
# Expected: Ollama OK
```

## File Locations

### Configuration Files
- `requirements.txt` - Python dependencies
- `Utilities.py` - Configuration and utilities
- `.gitignore` - Git ignore rules (includes chroma_db)

### Application Files
- `localragdemo.py` - Main application
- `audio_page.py` - Audio processing page
- `chat_page.py` - Chat interface
- `Utilities.py` - Shared utilities

### Data Directories
- `chroma_db/` - Vector database storage
- `logs/` - Application logs
- `split_pdfs/` - Temporary PDF splits
- `.venv/` - Python virtual environment

### Documentation Files
- `README.md` - Main documentation
- `AUDIO_GUIDE.md` - Audio processing guide
- `AUDIO_EXAMPLES.md` - Example use cases
- `AUDIO_FEATURE_SUMMARY.md` - Feature summary

### Scripts
- `install_audio_dependencies.ps1` - Automated setup
- `scripts/test_audio_setup.py` - Verify setup
- `scripts/check_chromadb.py` - Check ChromaDB
- `scripts/test_chroma_ollama.py` - Test integration

## Port and Access

### Default URLs
- Streamlit app: http://localhost:8501
- Network access: http://<your-ip>:8501

### Change Port
```powershell
streamlit run localragdemo.py --server.port 8080
```

### Open in Browser
```powershell
streamlit run localragdemo.py --server.headless false
```

## Backup Commands

### Backup ChromaDB
```powershell
# Create backup
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
Copy-Item -Recurse chroma_db "chroma_db_backup_$timestamp"
```

### Restore ChromaDB
```powershell
# List backups
Get-ChildItem -Filter "chroma_db_backup_*"

# Restore from backup
Remove-Item -Recurse -Force chroma_db
Copy-Item -Recurse chroma_db_backup_20250115_120000 chroma_db
```

## Performance Tips

### Monitor Resource Usage
```powershell
# Open Task Manager
taskmgr

# Or use PowerShell
Get-Process python | Select-Object CPU, WorkingSet, ProcessName
```

### Reduce Memory Usage
```powershell
# Use smaller models
$env:LLM_MODEL = "tinyllama"
$env:EMBEDDING_MODEL = "phi"

# Restart application
streamlit run localragdemo.py
```

## Getting Help

### View Documentation
```powershell
# Open guides in default editor
notepad AUDIO_GUIDE.md
notepad AUDIO_EXAMPLES.md
notepad README.md
```

### Check Logs
```powershell
# View latest log
Get-ChildItem logs | Sort-Object LastWriteTime -Descending | Select-Object -First 1 | Get-Content
```

### Run Diagnostics
```powershell
python scripts/test_audio_setup.py
```
