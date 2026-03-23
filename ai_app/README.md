## Steeves AI (Compute-first)

This app avoids notebook-state problems by:
- Loading your dataset from disk every run
- Computing KPI questions deterministically (pandas)
- Using your local Ollama only to explain computed results (optional)
- Regenerating interactive charts directly from the dataset (Plotly)

### Run

From the project folder:

```bash
python3 -m pip install -r requirements.txt
streamlit run ai_app/app.py
```

### Notes
- **No SQLite required** for your current size/use-case. Pandas is faster to iterate.
- SQLite can be useful later if you want ad-hoc SQL queries and persistent derived tables, but it’s optional.

### Files it uses (defaults)
- Dataset: `simulated_data (1).csv`
