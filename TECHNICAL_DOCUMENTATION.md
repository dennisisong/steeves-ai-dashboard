### Project objectives
This project delivers an interactive analytics dashboard and a question-driven assistant to help stakeholders understand revenue, profitability, utilization, and performance drivers across clients, projects, and consultant roles. The solution is designed to support repeatable KPI reporting and rapid exploration via filters and natural-language prompts.

### Business problem
Leadership and delivery stakeholders need a consistent way to answer questions such as:
- Which clients/projects/roles drive the most revenue?
- Which segments are most/least profitable?
- How does performance change over time?
- How do filters (time range, region, project type, billable-only) affect KPIs?

### Data
- **Raw dataset**: `simulated_data (1).csv`
- **Grain**: time-entry style records (client/project/consultant/date measures)
- **Key fields**: `Worked Date`, `Client Name`, `Project Name`, `Consultant Name`, `Consultant Role`, `Revenue`, `Gross Margin`, `Billable Hours`, `Client Region`, `Project Type`, (optional) satisfaction score fields.

### Libraries (and versions)
The app uses the dependencies in `requirements.txt`:
- `streamlit` (UI)
- `pandas`, `numpy` (data wrangling + aggregation)
- `plotly` (interactive charts)
- `requests` (Ollama HTTP calls)
- `faster-whisper` (optional voice transcription)

### Methods / algorithms used
**1) Compute-first analytics (deterministic KPIs)**
- KPI questions are answered via pandas groupby aggregations (source-of-truth).
- Examples:
  - Revenue by role: group by `Consultant Role`, sum `Revenue` (and optionally `Gross Margin`, `Billable Hours`)
  - Top clients/projects: group by entity, sum metrics, sort descending
  - Margin %: \( \text{Margin \%} = \frac{\text{Gross Margin}}{\text{Revenue}} \times 100 \)

**2) Visualization**
- Plotly charts are regenerated directly from the filtered dataset:
  - Monthly revenue trend (line chart)
  - Top-N rankings (horizontal bar charts)
  - Satisfaction vs margin (scatter) when satisfaction + margin are available
- Charts are interactive (hover, zoom, pan, selections) and reflect filter state.

**3) Optional LLM explanation layer (local)**
- Local Ollama is used optionally to explain computed results.
- The model is instructed to use only the computed table/known context (reducing hallucinations).

**4) Voice input (optional)**
- Streamlit audio recording is transcribed with Faster-Whisper on CPU.
- Transcript is inserted into the chat bar for review before sending.

### Solution overview (application)
Source code: `ai_app/app.py`
- **Tabs**: Dashboard / Chat / Data
- **Filters**: date range, client, role, region, project type, billable-only
- **Persistent chat storage**: `.steeves_chat_sessions/` (created at runtime)

### Key insights and recommendations (template)
Fill in based on your dashboard results:
- **Insight 1**: [e.g., revenue concentration among top clients/projects]
  - Recommendation: [account focus / retention / pricing]
- **Insight 2**: [e.g., role mix impacts margin]
  - Recommendation: [staffing changes, rate review]
- **Insight 3**: [e.g., certain project types show lower margin]
  - Recommendation: [scope control, delivery method changes]

### Limitations
- Dataset is limited to available fields and quality (missingness/outliers can affect aggregates).
- LLM explanations are optional and depend on the locally installed model; KPIs/charts remain deterministic.
- Real-time speech-to-text streaming is not implemented; transcription occurs after recording completes.

### Next steps
- Add more KPI tools (utilization, budget variance, invoice aging, risk).
- Add export (PNG/PDF for charts; slide-ready summaries).
- Add authentication / role-based views if deployed to a shared environment.

