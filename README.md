### Steeves & Associates — Analytics App + AI Assistant
This repository contains a Streamlit analytics application that:
- loads `simulated_data (1).csv`
- provides interactive charts and filters (Dashboard)
- answers common KPI questions deterministically (compute-first) and optionally uses local Ollama to explain results (Chat)
- supports optional speech-to-text for drafting prompts

### Key files
- **App**: `ai_app/app.py`
- **Dependencies**: `requirements.txt`
- **Setup**: `SETUP.md`
- **User guide**: `USER_GUIDE.md`
- **Technical documentation**: `TECHNICAL_DOCUMENTATION.md`
- **Slides outline**: `SLIDES_OUTLINE.md`
- **Deliverables checklist**: `DELIVERABLES.md`

### Run
```bash
python3 -m pip install -r requirements.txt
python3 -m streamlit run ai_app/app.py
```

