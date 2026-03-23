### Slide 1 — Title
**Steeves & Associates: Interactive Analytics Dashboard + AI Assistant**

- Course / Sponsor / Date
- Team name
- Team members

Speaker notes
- 1–2 sentences: what you built (dashboard + compute-first AI assistant) and what business question it answers.

---

### Slide 2 — Executive Summary
- **Company & context**: Steeves & Associates time-entry / project delivery analytics
- **Business problem**: understand revenue, profitability, utilization, and drivers across clients/projects/roles
- **Objective**: deliver an interactive dashboard + question-driven assistant for fast insight retrieval
- **Solution**: Streamlit analytics app with interactive charts, filters, and chat (compute-first + optional Ollama explanations)
- **Impact**: faster decision-making; consistent KPI reporting; easier stakeholder exploration

Speaker notes
- Keep to ~45–60 seconds.

---

### Slide 3 — Team Members & Roles
Fill in:
- **Name** — Data preparation / feature engineering
- **Name** — EDA + insights
- **Name** — Dashboard design + UX
- **Name** — AI assistant + app integration
- **Name** — Testing + documentation + demo script

Speaker notes
- Mention how you collaborated (branching, dividing work, weekly syncs, etc.).

---

### Slide 4 — Project Objectives & KPIs
**Primary goals**
- Surface drivers of **Revenue** and **Gross Margin**
- Identify top clients/projects/roles and profitability patterns
- Support operational questions (utilization, billable hours, satisfaction)

**Success metrics / KPIs**
- Total revenue, total gross margin, margin %
- Billable hours (and billable-only filtering)
- Top-N rankings (clients/projects/roles/consultants)
- Trends over time (monthly revenue)
- Satisfaction vs margin relationship (where available)

---

### Slide 5 — Data Overview
- Data source: `simulated_data (1).csv`
- Row/column counts
- Key fields: Client, Project, Consultant, Worked Date, Revenue, Gross Margin, Billable Hours, Region, Project Type, Satisfaction

Speaker notes
- Clarify this is simulated/project data (if applicable).

---

### Slide 6 — EDA Summary (high-signal findings)
Add 3–6 bullets, examples:
- Revenue concentration (top clients/projects contribute a large share)
- Margin variability by project type / role
- Utilization patterns by role/location (if used)
- Trend periods (months with spikes/dips)
- Satisfaction vs profitability observations (if used)

Speaker notes
- You can show 1–2 numbers and 1–2 charts; don’t list everything.

---

### Slide 7 — Key Visuals (EDA evidence)
Include screenshots of:
- Revenue trend (monthly)
- Top roles by revenue
- Top clients by revenue
- Satisfaction vs margin (if available)

Speaker notes
- Tie each chart to a business decision.

---

### Slide 8 — Analytics Solution (Use Case)
**Who uses it**
- Leadership / finance / delivery managers

**User journey**
- Pick filters (date, clients, roles, region, project type, billable-only)
- Review KPIs + charts (Dashboard)
- Ask questions in Chat (e.g., “top roles by revenue”, “plot revenue trend”)

**Value**
- Faster answers to recurring questions
- Fewer manual spreadsheets
- Consistent KPI definitions across stakeholders

---

### Slide 9 — Architecture (why it works reliably)
**Compute-first principle**
- Python/pandas computes KPIs deterministically (source of truth)
- The LLM (Ollama) is optional and only explains computed results

**Components**
- Streamlit UI (Dashboard / Chat / Data tabs)
- Plotly interactive charts
- Local Ollama (optional) for explanation layer
- Persistent chat storage (`.steeves_chat_sessions/`)

---

### Slide 10 — Live Demo Plan (2–4 minutes)
1. Open Dashboard and apply a filter (date/client/role)
2. Show interactive hover/zoom on a chart
3. Ask Chat: “Top 10 consultant roles by revenue”
4. Ask Chat: “Plot revenue trend”
5. (Optional) Use voice: record → transcript fills chat bar → send

---

### Slide 11 — Results & Business Impact
Include:
- Example: “Top roles by revenue” and how it informs staffing/pricing
- Example: “Top clients/projects” and how it supports account focus
- Example: “Margin % visibility” and how it supports profitability decisions

If you have estimates, add:
- Time saved vs manual reporting
- Faster turnaround for stakeholder questions

---

### Slide 12 — Challenges & Solutions
Use 3–5 bullets:
- Notebook AI context limitations → moved to compute-first app
- LLM generic answers → deterministic KPI computation + charts
- Embedding model availability → made Ollama optional
- Voice UX + state issues → used Streamlit audio recording + safer state flow

---

### Slide 13 — Next Steps & Recommendations
**Next steps**
- Add more KPI tools (utilization, budget variance, aging invoices, risk)
- Add export to PDF/PowerPoint for charts
- Role-based views (finance vs delivery)
- Deploy to internal server (optional)

**Recommendations**
- Focus on top margin drivers by role/project type
- Identify low-margin segments and improve pricing/scope control

---

### Slide 14 — Closing / Q&A
- What you built
- Why it matters
- Invite sponsor questions

