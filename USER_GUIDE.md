### What this is
This project ships a Streamlit analytics app that lets a stakeholder:
- Explore KPIs and interactive charts with filters
- Ask common business questions in Chat (with compute-first answers and optional LLM explanations)
- (Optional) use speech-to-text to draft chat prompts

### What you need installed
- **Python**: 3.10+ recommended
- **Ollama (optional)**: for AI explanations only (the dashboard/KPIs work without it)
- **ffmpeg (recommended)**: helps with audio decoding on some systems

### Setup (quick)
Follow `SETUP.md`.

### Run the app
From the project root:

```bash
python3 -m streamlit run ai_app/app.py
```

### Using the app
#### Dashboard tab
- **Filters**: choose date range and business filters (client/role/region/project type/billable-only)
- **Charts**: interact with charts (hover values, zoom, pan). All charts update with filters.

#### Chat tab
- **Ask KPI questions** (examples):
  - “Top consultant roles by revenue”
  - “Top clients by gross margin”
  - “Top projects by revenue”
- **Ask for charts** (examples):
  - “Plot revenue trend”
  - “Show top clients chart”
- **Clear chat**: use the “Clear chat” button to wipe the current session history.

#### Voice (speech-to-text)
- Record audio using the voice control in Chat.
- After recording completes, the transcript is placed into the chat input.
- Review/edit the transcript, then press **Enter** or **Send**.

### Demo script (for the live presentation)
1. **Dashboard**: apply a filter (e.g., a client or date range) and show charts updating.
2. **Chat (compute-first)**: ask “Top consultant roles by revenue” and show the table + chart result.
3. **Chat (chart request)**: ask “Plot revenue trend” and show the generated chart.
4. **Voice (optional)**: record “top clients by revenue”, confirm transcript, send.
5. **Clear chat**: click “Clear chat” to show clean reset for the next stakeholder.

### Troubleshooting
- **Streamlit not found**:
  - Run `python3 -m pip install -r requirements.txt`
- **Ollama not running** (only affects AI explanations):
  - Start Ollama and ensure a model is pulled (see `SETUP.md`)
- **Voice transcription slow**:
  - Shorten recordings; Faster-Whisper runs locally on CPU by default.
- **Audio issues**:
  - Install ffmpeg (macOS: `brew install ffmpeg`) and retry.

