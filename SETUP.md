## Setup (Run the Steeves AI app)

### 0) (Recommended) Install ffmpeg for voice input
If you plan to use speech-to-text, ffmpeg can help with audio decoding on some systems.

- macOS (Homebrew):

```bash
brew install ffmpeg
```

### 1) Install Ollama (optional but recommended)
- Install Ollama for macOS from `https://ollama.com`
- Start Ollama (it runs a local server on `http://localhost:11434`)

Pull a chat model (one example):

```bash
ollama pull llama3.2:3b
```

> If you don’t want Ollama, you can still run the app and use **charts + computed KPIs** (just turn off “Use Ollama for explanations” in the sidebar).

### 2) Install Python deps

From this folder:

```bash
python3 -m pip install -r requirements.txt
```

### 3) Run the app

```bash
python3 -m streamlit run ai_app/app.py
```

Streamlit will print a local URL (usually `http://localhost:8501`).

### 4) Notes
- The app will create a local folder `.steeves_chat_sessions/` automatically to persist chat history.
