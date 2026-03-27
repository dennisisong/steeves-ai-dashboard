import os
import re
import base64
import glob
import json
import uuid
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import requests
import streamlit as st
from openai import OpenAI

try:
    from faster_whisper import WhisperModel
except Exception:
    WhisperModel = None


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATASET = os.path.join(PROJECT_DIR, "simulated_data (1).csv")
DEFAULT_LOGO = os.path.join(PROJECT_DIR, "NU_RGB_seal_R.png")


def _find_first(pattern: str) -> str:
    matches = sorted(glob.glob(pattern))
    return matches[0] if matches else ""


DEFAULT_SIDE_ART = _find_first(os.path.join(PROJECT_DIR, "Screenshot*.png"))


def _img_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def inject_branding(logo_path: str, side_art_path: str = "") -> None:
    st.markdown(
        f"""
        <style>
            .stApp {{
            background-color: #001f3f;
            color: #ffffff;
            }}

            /* Sidebar */
            section[data-testid="stSidebar"] > div:first-child {{
            background: rgba(255, 255, 255, 0.08);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border-right: 1px solid rgba(255,255,255,0.1);
            }}

            /* Chat bubbles */
            div[data-testid="stChatMessage"] {{
            background: rgba(255, 255, 255, 0.06);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 14px;
            padding: 0.25rem 0.75rem;
            box-shadow: 0 14px 40px rgba(0,0,0,0.25);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            }}

            /* Charts container */
            div[data-testid="stPlotlyChart"] {{
            background: rgba(255, 255, 255, 0.06);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 14px;
            padding: 0.25rem 0.5rem;
            box-shadow: 0 14px 40px rgba(0,0,0,0.25);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            }}

            /* Side artwork */
            .steeves-sidecard {{
            border-radius: 18px;
            border: 1px solid rgba(255,255,255,0.08);
            background: rgba(255,255,255,0.06);
            backdrop-filter: blur(14px);
            -webkit-backdrop-filter: blur(14px);
            box-shadow: 0 20px 60px rgba(0,0,0,0.25);
            overflow: hidden;
            margin: 0.25rem 0 0.75rem 0;
            }}

            .steeves-sidecard img {{
            width: 100%;
            display: block;
            opacity: 0.28;
            filter: saturate(1.15) contrast(1.03);
            }}

            /* Headings */
            h1, h2, h3, h4 {{
            color: #ffffff;
            }}

            /* Buttons */
            button[kind="primary"] {{
            background-color: #0a84ff;
            border-radius: 12px;
            border: none;
            font-weight: 500;
            }}

            button[kind="primary"]:hover {{
            background-color: #0066cc;
            }}

            /* Padding */
            .block-container {{
            padding-top: 1.25rem;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Sidebar logo (optional)
    if logo_path and os.path.exists(logo_path):
        # keep sidebar clean; don't render logo there
        pass


def normalize_host(raw: str) -> str:
    s = (raw or "").strip()
    s = s.lstrip(" ?/\t\r\n").replace(" ", "")
    if not s:
        s = "http://localhost:11434"
    if not (s.startswith("http://") or s.startswith("https://")):
        s = "http://" + s
    return s.rstrip("/")


def ollama_list_models(host: str, timeout: int = 5) -> List[str]:
    host = normalize_host(host)
    r = requests.get(f"{host}/api/tags", timeout=timeout)
    r.raise_for_status()
    data = r.json()
    return [m.get("name") for m in data.get("models", []) if m.get("name")]


def ollama_chat(host: str, model: str, messages: List[Dict[str, str]], timeout: int = 120) -> str:
    host = normalize_host(host)
    r = requests.post(
        f"{host}/api/chat",
        json={"model": model, "messages": messages, "stream": False, "options": {"temperature": 0.2}},
        timeout=timeout,
    )
    r.raise_for_status()
    return r.json()["message"]["content"]


@st.cache_data(show_spinner=False)
def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # best-effort parse dates if present
    for col in ["Worked Date", "Estimated Completion Date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def with_margin_pct(frame: pd.DataFrame, rev_col: str = "Revenue", gm_col: str = "Gross Margin") -> pd.DataFrame:
    frame = frame.copy()
    if rev_col in frame.columns and gm_col in frame.columns:
        denom = frame[rev_col].where(frame[rev_col] != 0)
        frame["Margin_%"] = (frame[gm_col] / denom) * 100
    return frame


def compute_revenue_by_role(df: pd.DataFrame, top_n: int = 10) -> Optional[pd.DataFrame]:
    if "Consultant Role" not in df.columns or "Revenue" not in df.columns:
        return None
    out = df.groupby("Consultant Role", dropna=False)[["Revenue"]].sum(numeric_only=True).reset_index()
    if "Gross Margin" in df.columns:
        gm = df.groupby("Consultant Role", dropna=False)[["Gross Margin"]].sum(numeric_only=True).reset_index()
        out = out.merge(gm, on="Consultant Role", how="left")
        out = with_margin_pct(out)
    if "Billable Hours" in df.columns:
        bh = df.groupby("Consultant Role", dropna=False)[["Billable Hours"]].sum(numeric_only=True).reset_index()
        out = out.merge(bh, on="Consultant Role", how="left")
    return out.sort_values("Revenue", ascending=False).head(int(top_n))


def compute_top_consultants(df: pd.DataFrame, top_n: int = 10) -> Optional[pd.DataFrame]:
    if "Consultant Name" not in df.columns:
        return None
    if "Revenue" in df.columns:
        out = df.groupby("Consultant Name", dropna=False)[["Revenue"]].sum(numeric_only=True).reset_index()
        if "Gross Margin" in df.columns:
            gm = df.groupby("Consultant Name", dropna=False)[["Gross Margin"]].sum(numeric_only=True).reset_index()
            out = out.merge(gm, on="Consultant Name", how="left")
            out = with_margin_pct(out)
        if "Billable Hours" in df.columns:
            bh = df.groupby("Consultant Name", dropna=False)[["Billable Hours"]].sum(numeric_only=True).reset_index()
            out = out.merge(bh, on="Consultant Name", how="left")
        return out.sort_values("Revenue", ascending=False).head(int(top_n))
    if "Billable Hours" in df.columns:
        out = df.groupby("Consultant Name", dropna=False)[["Billable Hours"]].sum(numeric_only=True).reset_index()
        return out.sort_values("Billable Hours", ascending=False).head(int(top_n))
    return None


def compute_top_clients(df: pd.DataFrame, top_n: int = 10) -> Optional[pd.DataFrame]:
    if "Client Name" not in df.columns or "Revenue" not in df.columns:
        return None
    out = df.groupby("Client Name", dropna=False)[["Revenue"]].sum(numeric_only=True).reset_index()
    if "Gross Margin" in df.columns:
        gm = df.groupby("Client Name", dropna=False)[["Gross Margin"]].sum(numeric_only=True).reset_index()
        out = out.merge(gm, on="Client Name", how="left")
        out = with_margin_pct(out)
    if "Billable Hours" in df.columns:
        bh = df.groupby("Client Name", dropna=False)[["Billable Hours"]].sum(numeric_only=True).reset_index()
        out = out.merge(bh, on="Client Name", how="left")
    return out.sort_values("Revenue", ascending=False).head(int(top_n))


def compute_top_projects(df: pd.DataFrame, top_n: int = 10) -> Optional[pd.DataFrame]:
    if "Project Name" not in df.columns or "Revenue" not in df.columns:
        return None
    out = df.groupby("Project Name", dropna=False)[["Revenue"]].sum(numeric_only=True).reset_index()
    if "Gross Margin" in df.columns:
        gm = df.groupby("Project Name", dropna=False)[["Gross Margin"]].sum(numeric_only=True).reset_index()
        out = out.merge(gm, on="Project Name", how="left")
        out = with_margin_pct(out)
    return out.sort_values("Revenue", ascending=False).head(int(top_n))


def chart_revenue_trend(df: pd.DataFrame) -> Optional[Any]:
    if "Worked Date" not in df.columns or "Revenue" not in df.columns:
        return None
    d = df.dropna(subset=["Worked Date"]).copy()
    if d.empty:
        return None
    d["Month"] = d["Worked Date"].dt.to_period("M").dt.to_timestamp()
    m = d.groupby("Month", dropna=False)[["Revenue"]].sum(numeric_only=True).reset_index()
    m = m.sort_values("Month")
    fig = px.line(m, x="Month", y="Revenue", title="Revenue trend over time (monthly)", markers=True)
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=50, b=10))
    return fig


def chart_top_roles(df: pd.DataFrame, top_n: int = 10) -> Optional[Any]:
    t = compute_revenue_by_role(df, top_n=top_n)
    if t is None or t.empty:
        return None
    fig = px.bar(t, x="Revenue", y="Consultant Role", orientation="h", title=f"Top {top_n} roles by revenue")
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=50, b=10), yaxis=dict(categoryorder="total ascending"))
    return fig


def chart_top_clients(df: pd.DataFrame, top_n: int = 10) -> Optional[Any]:
    t = compute_top_clients(df, top_n=top_n)
    if t is None or t.empty:
        return None
    fig = px.bar(t, x="Revenue", y="Client Name", orientation="h", title=f"Top {top_n} clients by revenue")
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=50, b=10), yaxis=dict(categoryorder="total ascending"))
    return fig


def chart_top_projects(df: pd.DataFrame, top_n: int = 10) -> Optional[Any]:
    t = compute_top_projects(df, top_n=top_n)
    if t is None or t.empty:
        return None
    fig = px.bar(t, x="Revenue", y="Project Name", orientation="h", title=f"Top {top_n} projects by revenue")
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=50, b=10), yaxis=dict(categoryorder="total ascending"))
    return fig


def chart_margin_vs_satisfaction(df: pd.DataFrame) -> Optional[Any]:
    if "Client Name" not in df.columns or "Revenue" not in df.columns:
        return None

    sat_col = None
    for c in ["Client Satisfaction Score", "Client Satisfaction Score_y", "Client Satisfaction Score_x"]:
        if c in df.columns:
            sat_col = c
            break
    if sat_col is None:
        return None

    m = df.groupby("Client Name", dropna=False).agg(
        Revenue=("Revenue", "sum"),
        Gross_Margin=("Gross Margin", "sum") if "Gross Margin" in df.columns else ("Revenue", "sum"),
        Satisfaction=(sat_col, "first"),
    ).reset_index()

    if "Gross Margin" in df.columns:
        m = m.rename(columns={"Gross_Margin": "Gross Margin"})
        m = with_margin_pct(m)
        y = "Margin_%"
        y_title = "Margin %"
    else:
        return None

    m = m.dropna(subset=["Satisfaction", y])
    if m.empty:
        return None

    fig = px.scatter(
        m,
        x="Satisfaction",
        y=y,
        size="Revenue",
        hover_name="Client Name",
        title="Client satisfaction vs profitability",
        size_max=60,
    )
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=50, b=10))
    return fig


def chart_role_profitability(df: pd.DataFrame, top_n: int = 15) -> Optional[Any]:
    """Revenue (bar) + margin % (line) by consultant role."""
    required = {"Consultant Role", "Revenue", "Gross Margin"}
    if not required.issubset(set(df.columns)):
        return None

    role = (
        df.groupby("Consultant Role", dropna=False)
        .agg(
            Revenue=("Revenue", "sum"),
            Gross_Margin=("Gross Margin", "sum"),
        )
        .reset_index()
    )

    role = role.rename(columns={"Gross_Margin": "Gross Margin"})
    role = with_margin_pct(role)
    role = role.sort_values("Revenue", ascending=False).head(int(top_n))

    if role.empty:
        return None

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(
            name="Revenue",
            x=role["Consultant Role"],
            y=role["Revenue"],
            marker_color="#3498db",
            hovertemplate="Role=%{x}<br>Revenue=%{y:$,.0f}<extra></extra>",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            name="Margin %",
            x=role["Consultant Role"],
            y=role["Margin_%"],
            mode="lines+markers",
            marker_color="#e74c3c",
            line=dict(width=3, color="#e74c3c"),
            hovertemplate="Role=%{x}<br>Margin=%{y:.1f}%<extra></extra>",
        ),
        secondary_y=True,
    )

    fig.update_layout(
        title="Revenue & Profitability by Consultant Role",
        height=480,
        hovermode="x unified",
        margin=dict(l=10, r=10, t=50, b=10),
    )
    fig.update_xaxes(title_text="Role", tickangle=-45)
    fig.update_yaxes(title_text="Revenue ($)", secondary_y=False)
    fig.update_yaxes(title_text="Margin (%)", secondary_y=True)
    return fig


def chart_utilization_by_role(df: pd.DataFrame, top_n: int = 15) -> Optional[Any]:
    if "Consultant Role" not in df.columns or "Billable Hours" not in df.columns:
        return None
    m = (
        df.groupby("Consultant Role", dropna=False)["Billable Hours"]
        .sum(numeric_only=True)
        .reset_index()
    )
    m = m.sort_values("Billable Hours", ascending=False).head(int(top_n))
    if m.empty:
        return None
    fig = px.bar(
        m,
        x="Billable Hours",
        y="Consultant Role",
        orientation="h",
        title=f"Total Billable Hours by Role (Top {top_n})",
        color="Billable Hours",
        color_continuous_scale="Viridis",
    )
    fig.update_layout(height=440, margin=dict(l=10, r=10, t=50, b=10))
    return fig


def chart_utilization_by_location(df: pd.DataFrame, top_n: int = 15) -> Optional[Any]:
    if "Consultant Location" not in df.columns or "Billable Hours" not in df.columns:
        return None
    m = (
        df.groupby("Consultant Location", dropna=False)["Billable Hours"]
        .sum(numeric_only=True)
        .reset_index()
    )
    m = m.sort_values("Billable Hours", ascending=False).head(int(top_n))
    if m.empty:
        return None
    fig = px.bar(
        m,
        x="Billable Hours",
        y="Consultant Location",
        orientation="h",
        title=f"Total Billable Hours by Location (Top {top_n})",
        color="Billable Hours",
        color_continuous_scale="Viridis",
    )
    fig.update_layout(height=440, margin=dict(l=10, r=10, t=50, b=10))
    return fig


def chart_project_timeline(df: pd.DataFrame, top_n: int = 15) -> Optional[Any]:
    required = {"Project Name", "Worked Date", "Revenue", "Project Type"}
    if not required.issubset(set(df.columns)):
        return None

    d = df.dropna(subset=["Worked Date"]).copy()
    if d.empty:
        return None

    proj = (
        d.groupby("Project Name", dropna=False)
        .agg(
            Start=("Worked Date", "min"),
            End=("Worked Date", "max"),
            Revenue=("Revenue", "sum"),
            Gross_Margin=("Gross Margin", "sum") if "Gross Margin" in d.columns else ("Revenue", "sum"),
            Project_Type=("Project Type", "first"),
        )
        .reset_index()
    )
    proj = proj.rename(columns={"Gross_Margin": "Gross Margin", "Project_Type": "Project Type"})

    proj["Duration_Days"] = (proj["End"] - proj["Start"]).dt.days
    proj = proj.sort_values("Revenue", ascending=False).head(int(top_n))
    if proj.empty:
        return None

    fig = px.timeline(
        proj,
        x_start="Start",
        x_end="End",
        y="Project Name",
        color="Project Type",
        title=f"Top {top_n} Projects Timeline",
        labels={"Project Name": "Project"},
    )
    fig.update_yaxes(categoryorder="total ascending")
    fig.update_layout(height=560, margin=dict(l=10, r=10, t=50, b=10))
    return fig


def chart_top_projects_by_gross_margin(df: pd.DataFrame, top_n: int = 10) -> Optional[Any]:
    if "Project Name" not in df.columns or "Gross Margin" not in df.columns or "Revenue" not in df.columns:
        return None

    m = (
        df.groupby("Project Name", dropna=False)
        .agg(
            Revenue=("Revenue", "sum"),
            Gross_Margin=("Gross Margin", "sum"),
        )
        .reset_index()
    )

    m["Margin_Pct"] = np.where(m["Revenue"] != 0, (m["Gross_Margin"] / m["Revenue"]) * 100.0, 0.0)
    m = m.sort_values("Gross_Margin", ascending=False).head(int(top_n))
    if m.empty:
        return None

    fig = px.bar(
        m,
        x="Gross_Margin",
        y="Project Name",
        orientation="h",
        title=f"Top {top_n} Projects by Gross Margin",
        color="Margin_Pct",
        color_continuous_scale="RdYlGn",
        hover_data={"Revenue": ":,.0f", "Margin_Pct": ":.1f"},
    )
    fig.update_layout(height=460, margin=dict(l=10, r=10, t=50, b=10))
    fig.update_yaxes(categoryorder="total ascending")
    return fig


def chart_project_type_compare(df: pd.DataFrame) -> Optional[Any]:
    required = {"Project Type", "Revenue", "Gross Margin"}
    if not required.issubset(set(df.columns)):
        return None

    m = (
        df.groupby("Project Type", dropna=False)
        .agg(Revenue=("Revenue", "sum"), Gross_Margin=("Gross Margin", "sum"))
        .reset_index()
    )
    m = m.sort_values("Revenue", ascending=False)
    if m.empty:
        return None

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Revenue", x=m["Project Type"], y=m["Revenue"], marker_color="#3498db"))
    fig.add_trace(
        go.Bar(name="Gross Margin", x=m["Project Type"], y=m["Gross_Margin"], marker_color="#2ecc71")
    )
    fig.update_layout(
        title="Revenue vs Gross Margin by Project Type",
        barmode="group",
        height=520,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    fig.update_xaxes(tickangle=-45)
    return fig


def chart_top_consultants(df: pd.DataFrame, top_n: int = 10) -> Optional[Any]:
    t = compute_top_consultants(df, top_n=top_n)
    if t is None or t.empty:
        return None
    metric = "Revenue" if "Revenue" in t.columns else ("Billable Hours" if "Billable Hours" in t.columns else None)
    if metric is None:
        return None
    fig = px.bar(
        t,
        x=metric,
        y="Consultant Name",
        orientation="h",
        title=f"Top {top_n} Consultants by {metric}",
        color=metric,
        color_continuous_scale="Blues",
    )
    fig.update_layout(height=460, margin=dict(l=10, r=10, t=50, b=10), yaxis={"categoryorder": "total ascending"})
    return fig


def parse_chart_intent(q: str) -> Optional[Tuple[str, int]]:
    s = (q or "").lower()
    wants = any(k in s for k in ["plot", "chart", "graph", "visual", "show me", "draw"])
    if not wants:
        return None
    top_n = parse_top_n(s, 10)
    if "trend" in s and "revenue" in s:
        return ("revenue_trend", top_n)
    if ("role" in s or "roles" in s) and ("margin" in s or "profit" in s) and ("revenue" in s or "profit" in s or "profitability" in s):
        return ("role_profitability", top_n)
    if ("role" in s or "roles" in s) and "revenue" in s:
        return ("top_roles", top_n)
    if "client" in s and "revenue" in s:
        return ("top_clients", top_n)
    if "project" in s and "revenue" in s:
        return ("top_projects", top_n)
    if ("consultant" in s or "consultants" in s) and "revenue" in s and "role" not in s:
        return ("top_consultants", top_n)
    if "timeline" in s and "project" in s:
        return ("project_timeline", top_n)
    if ("gross margin" in s or ("margin" in s and "gross" in s) or "profitability" in s) and "project" in s:
        return ("top_projects_gross_margin", top_n)
    if "project type" in s or ("type" in s and "project" in s):
        return ("project_type", top_n)
    if ("utilization" in s or "utilisation" in s) and "location" in s:
        return ("utilization_by_location", top_n)
    if ("utilization" in s or "utilisation" in s) and "role" in s:
        return ("utilization_by_role", top_n)
    if "satisfaction" in s and ("margin" in s or "profit" in s or "profitability" in s):
        return ("satisfaction_margin", top_n)
    return None


def parse_top_n(q: str, default: int = 10) -> int:
    m = re.search(r"\btop\s+(\d+)\b", (q or "").lower())
    n = int(m.group(1)) if m else default
    return max(1, min(n, 50))


def compute_intent(q: str, df: pd.DataFrame) -> Optional[Tuple[str, pd.DataFrame, str]]:
    s = (q or "").lower()
    top_n = parse_top_n(s, 10)

    # roles
    if ("role" in s or "roles" in s) and ("revenue" in s or "most" in s or "highest" in s or "generate" in s):
        t = compute_revenue_by_role(df, top_n=top_n)
        if t is not None:
            return (f"Top {top_n} Consultant Roles by Revenue", t, "computed")

    # consultants
    if "consultant" in s and "role" not in s:
        t = compute_top_consultants(df, top_n=top_n)
        if t is not None:
            metric = "Revenue" if "Revenue" in t.columns else "Billable Hours"
            return (f"Top {top_n} Consultants by {metric}", t, "computed")

    # clients
    if "client" in s and ("revenue" in s or "top" in s or "most" in s or "highest" in s):
        t = compute_top_clients(df, top_n=top_n)
        if t is not None:
            return (f"Top {top_n} Clients by Revenue", t, "computed")

    # projects
    if "project" in s and ("revenue" in s or "top" in s or "most" in s or "highest" in s):
        t = compute_top_projects(df, top_n=top_n)
        if t is not None:
            return (f"Top {top_n} Projects by Revenue", t, "computed")

    return None


def _bar_chart_from_table(table: pd.DataFrame, title: str) -> Optional[Any]:
    if table is None or table.empty:
        return None

    cols = set(table.columns)
    metric = "Revenue" if "Revenue" in cols else ("Billable Hours" if "Billable Hours" in cols else None)
    if metric is None:
        return None

    if "Consultant Role" in cols:
        y = "Consultant Role"
    elif "Consultant Name" in cols:
        y = "Consultant Name"
    elif "Client Name" in cols:
        y = "Client Name"
    elif "Project Name" in cols:
        y = "Project Name"
    else:
        return None

    fig = px.bar(table, x=metric, y=y, orientation="h", title=title)
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=50, b=10), yaxis=dict(categoryorder="total ascending"))
    return fig


def df_schema_context(df: pd.DataFrame, max_cols: int = 60) -> str:
    cols = list(df.columns)
    head = df.head(5)
    # keep it compact
    return (
        f"DATASET SCHEMA:\n"
        f"- rows={len(df)}, cols={len(cols)}\n"
        f"- columns={cols[:max_cols]}{' ...' if len(cols) > max_cols else ''}\n\n"
        f"DATASET SAMPLE (first 5 rows):\n{head.to_markdown(index=False)}"
    )


# ----------------------------
# Chat sessions (persisted)
# ----------------------------

SESSIONS_DIR = Path(PROJECT_DIR) / ".steeves_chat_sessions"
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _session_path(session_id: str) -> Path:
    return SESSIONS_DIR / f"{session_id}.json"


def list_sessions() -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for p in sorted(SESSIONS_DIR.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(data, dict) and data.get("id"):
                out.append(
                    {
                        "id": data.get("id"),
                        "name": data.get("name") or "Untitled",
                        "created_at": data.get("created_at") or "",
                        "updated_at": data.get("updated_at") or "",
                    }
                )
        except Exception:
            continue
    return out


def load_session(session_id: str) -> Dict[str, Any]:
    data = json.loads(_session_path(session_id).read_text(encoding="utf-8"))
    if "messages" not in data or not isinstance(data["messages"], list):
        data["messages"] = []
    return data


def save_session(data: Dict[str, Any]) -> None:
    data = dict(data)
    data.setdefault("id", str(uuid.uuid4()))
    data.setdefault("name", "Chat")
    data.setdefault("created_at", _now_iso())
    data["updated_at"] = _now_iso()
    if "messages" not in data or not isinstance(data["messages"], list):
        data["messages"] = []
    _session_path(data["id"]).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def new_session(name: str = "New chat") -> Dict[str, Any]:
    data = {"id": str(uuid.uuid4()), "name": name, "created_at": _now_iso(), "updated_at": _now_iso(), "messages": []}
    save_session(data)
    return data


def delete_session(session_id: str) -> None:
    try:
        _session_path(session_id).unlink(missing_ok=True)
    except Exception:
        pass


# ----------------------------
# Filtering helpers
# ----------------------------

def apply_filters(df: pd.DataFrame, f: Dict[str, Any]) -> pd.DataFrame:
    out = df

    if "Worked Date" in out.columns and f.get("date_range"):
        start, end = f["date_range"]
        if start is not None:
            out = out[out["Worked Date"] >= pd.to_datetime(start)]
        if end is not None:
            out = out[out["Worked Date"] <= pd.to_datetime(end)]

    for col_key, col_name in [
        ("clients", "Client Name"),
        ("roles", "Consultant Role"),
        ("regions", "Client Region"),
        ("project_types", "Project Type"),
    ]:
        vals = f.get(col_key) or []
        if vals and col_name in out.columns:
            out = out[out[col_name].isin(vals)]

    if f.get("billable_only") and "Billable Flag" in out.columns:
        out = out[out["Billable Flag"].astype(str).str.lower().isin(["yes", "true", "1"])]

    return out


# ----------------------------
# Speech-to-text (optional)
# ----------------------------

@st.cache_resource(show_spinner=False)
def _get_whisper(model_size: str):
    if WhisperModel is None:
        raise RuntimeError("faster-whisper is not installed")
    # cpu-only by default (works everywhere)
    return WhisperModel(model_size, device="cpu", compute_type="int8")


def transcribe_audio_bytes(audio_bytes: bytes, suffix: str = ".wav", model_size: str = "base") -> str:
    if not audio_bytes:
        return ""
    suffix = suffix if (suffix and suffix.startswith(".")) else ".wav"
    model = _get_whisper(model_size)
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as f:
        f.write(audio_bytes)
        f.flush()
        segments, _info = model.transcribe(f.name, vad_filter=True)
        text = " ".join((seg.text or "").strip() for seg in segments).strip()
        return text


st.set_page_config(page_title="Steeves AI (Compute-first)", layout="wide", initial_sidebar_state="collapsed")
inject_branding(logo_path=DEFAULT_LOGO, side_art_path=DEFAULT_SIDE_ART)
pio.templates.default = "plotly_white"

# Header
header_l, header_mid, header_r = st.columns([0.08, 0.72, 0.20], vertical_alignment="center")

with header_l:
    if os.path.exists(DEFAULT_LOGO):
        st.image(DEFAULT_LOGO, width=60)

with header_mid:
    st.title("Steeves & Associates — AI Analytics")
    st.caption("Dashboard • Chat • Data")

with header_r:
    if DEFAULT_SIDE_ART and os.path.exists(DEFAULT_SIDE_ART):
        st.image(DEFAULT_SIDE_ART, use_container_width=True)

## add open ai
def ask_openai(prompt):
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content

# Defaults
dataset_path = os.environ.get("DATASET_PATH", DEFAULT_DATASET)
compute_first_default = os.environ.get("COMPUTE_FIRST", "1") not in {"0", "false", "False"}
use_ollama_default = False
host_default = normalize_host(os.environ.get("OLLAMA_HOST", "http://localhost:11434"))
model_default = os.environ.get("OLLAMA_MODEL", "llama3.2:3b")

if "settings" not in st.session_state:
    st.session_state.settings = {
        "compute_first": compute_first_default,
        "use_ollama": use_ollama_default,
        "host": host_default,
        "model": model_default,
    }

if "filters" not in st.session_state:
    st.session_state.filters = {
        "date_range": None,
        "clients": [],
        "roles": [],
        "regions": [],
        "project_types": [],
        "billable_only": False,
    }

# Sessions init
if "session_id" not in st.session_state:
    st.session_state.session_id = "default"
    if not _session_path("default").exists():
        save_session({"id": "default", "name": "Chat", "messages": []})

def _current_session() -> Dict[str, Any]:
    return load_session(st.session_state.session_id)

# Shared prompt handler (used by chat input + voice input)
def handle_prompt(prompt: str, current: Dict[str, Any]) -> None:
    if not prompt:
        return

    # append user
    current["messages"].append({"role": "user", "content": prompt})

    df = apply_filters(df_raw, st.session_state.filters)
    settings = st.session_state.settings

    # chart intent
    chart_req = parse_chart_intent(prompt)
    if chart_req is not None:
        kind, top_n = chart_req
        fig = None
        title = ""
        if kind == "revenue_trend":
            title = "Revenue trend (monthly)"
            fig = chart_revenue_trend(df)
        elif kind == "role_profitability":
            title = f"Revenue & Profitability by Role (Top {top_n})"
            fig = chart_role_profitability(df, top_n=top_n)
        elif kind == "top_roles":
            title = f"Top {top_n} roles by revenue"
            fig = chart_top_roles(df, top_n=top_n)
        elif kind == "top_consultants":
            title = f"Top {top_n} consultants by revenue"
            fig = chart_top_consultants(df, top_n=top_n)
        elif kind == "top_clients":
            title = f"Top {top_n} clients by revenue"
            fig = chart_top_clients(df, top_n=top_n)
        elif kind == "top_projects":
            title = f"Top {top_n} projects by revenue"
            fig = chart_top_projects(df, top_n=top_n)
        elif kind == "top_projects_gross_margin":
            title = f"Top {top_n} projects by gross margin"
            fig = chart_top_projects_by_gross_margin(df, top_n=top_n)
        elif kind == "project_timeline":
            title = f"Top {top_n} projects timeline"
            fig = chart_project_timeline(df, top_n=top_n)
        elif kind == "project_type":
            title = "Revenue vs Gross Margin by Project Type"
            fig = chart_project_type_compare(df)
        elif kind == "utilization_by_role":
            title = f"Utilization by Role (Top {top_n})"
            fig = chart_utilization_by_role(df, top_n=top_n)
        elif kind == "utilization_by_location":
            title = f"Utilization by Location (Top {top_n})"
            fig = chart_utilization_by_location(df, top_n=top_n)
        elif kind == "satisfaction_margin":
            title = "Client satisfaction vs margin"
            fig = chart_margin_vs_satisfaction(df)

        if fig is not None:
            current["messages"].append({"role": "assistant", "type": "plot", "content": f"**{title}**", "fig_json": fig.to_json()})
        else:
            current["messages"].append({"role": "assistant", "content": "I can’t build that chart from this dataset (missing required columns)."})

        save_session(current)
        st.rerun()

    # compute-first KPI
    computed = compute_intent(prompt, df) if settings["compute_first"] else None
    if computed is not None:
        title, table, _ = computed
        fig = _bar_chart_from_table(table, title=title)
        if fig is not None:
            current["messages"].append({"role": "assistant", "type": "plot", "content": f"**{title}**", "fig_json": fig.to_json()})
        else:
            current["messages"].append({"role": "assistant", "content": f"**{title}**\n\n(Computed, but no supported chart type for this result.)"})

        if settings["use_ollama"]:
            try:
                table_md = table.to_markdown(index=False)
                explanation = ollama_chat(
                    host=settings["host"],
                    model=settings["model"],
                    messages=[
                        {"role": "system", "content": "Use ONLY the provided table to answer. Do not invent numbers."},
                        {"role": "user", "content": f"Question: {prompt}\n\nTable:\n{table_md}\n\nExplain briefly."},
                    ],
                )
                current["messages"].append({"role": "assistant", "content": explanation})
            except Exception:
                current["messages"].append({"role": "assistant", "content": "(Ollama explanation failed; chart shown above.)"})

        save_session(current)
        st.rerun()

    # general LLM answer
           if not settings["use_ollama"]:
            try:
                reply = ask_openai(prompt)
        
                current["messages"].append({
                    "role": "assistant",
                    "content": reply
                })
        
            except Exception as e:
                current["messages"].append({
                    "role": "assistant",
                    "content": f"OpenAI error: {str(e)}"
                })
        
            save_session(current)
            st.rerun()

    ctx = df_schema_context(df)
    sys = (
        "You are a project assistant for this dataset.\n"
        "Answer using the provided context. If it's not in context, say you don't have it.\n"
        "Be concise and specific."
    )
    try:
        answer = ollama_chat(
            host=settings["host"],
            model=settings["model"],
            messages=[{"role": "system", "content": sys}, {"role": "user", "content": f"{ctx}\n\nQuestion:\n{prompt}"}],
        )
        current["messages"].append({"role": "assistant", "content": answer})
    except Exception as e:
        current["messages"].append({"role": "assistant", "content": f"Chat failed: {type(e).__name__}: {e}"})
    save_session(current)
    st.rerun()

# Settings (collapsed)
with st.expander("Settings", expanded=False):
    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        st.session_state.settings["compute_first"] = st.toggle("Compute-first", value=st.session_state.settings["compute_first"])
        st.session_state.settings["use_ollama"] = st.toggle("Ollama explanations", value=st.session_state.settings["use_ollama"])
        st.session_state.settings.setdefault("voice_auto_send", False)
        st.session_state.settings.setdefault("voice_model", "base")
        st.session_state.settings["voice_auto_send"] = st.toggle("Auto-send after transcription", value=st.session_state.settings["voice_auto_send"])
    with c2:
        st.session_state.settings["host"] = st.text_input("Host", value=st.session_state.settings["host"])
        st.session_state.settings["model"] = st.text_input("Model", value=st.session_state.settings["model"])
        st.session_state.settings["voice_model"] = st.selectbox("Voice model", options=["tiny", "base", "small"], index=["tiny","base","small"].index(st.session_state.settings["voice_model"]))
    with c3:
        if st.button("Test Ollama"):
            try:
                models = ollama_list_models(st.session_state.settings["host"])
                st.success(f"OK ({len(models)} model(s))")
            except Exception as e:
                st.error(f"Failed: {type(e).__name__}: {e}")

# Load dataset once
try:
    df_raw = load_csv(dataset_path)
except Exception as e:
    st.error(f"Dataset load failed: {type(e).__name__}: {e}")
    st.stop()

tabs = st.tabs(["Dashboard", "Chat", "Data"])

# ---------------- Dashboard ----------------
with tabs[0]:
    st.markdown(
        """
        <style>
        /* Dashboard tab only: first tab panel */
        div[role="tabpanel"]:first-of-type{
            background-color: #001f3f;
            border-radius: 18px;
            padding: 16px;
        }
        /* Keep headings readable */
        div[role="tabpanel"]:first-of-type h1,
        div[role="tabpanel"]:first-of-type h2,
        div[role="tabpanel"]:first-of-type h3,
        div[role="tabpanel"]:first-of-type h4{
            color: #ffffff !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.subheader("Dashboard")

    with st.expander("Filters", expanded=False):
        f = st.session_state.filters
        c1, c2 = st.columns([1, 1])
        with c1:
            if "Worked Date" in df_raw.columns and df_raw["Worked Date"].notna().any():
                min_d = df_raw["Worked Date"].min().date()
                max_d = df_raw["Worked Date"].max().date()
                f["date_range"] = st.date_input("Worked date range", value=(min_d, max_d))
            f["billable_only"] = st.toggle("Billable only", value=bool(f.get("billable_only", False)))
        with c2:
            if "Client Name" in df_raw.columns:
                opts = sorted([x for x in df_raw["Client Name"].dropna().unique().tolist()])[:500]
                f["clients"] = st.multiselect("Clients", options=opts, default=f.get("clients", []))
            if "Consultant Role" in df_raw.columns:
                opts = sorted([x for x in df_raw["Consultant Role"].dropna().unique().tolist()])
                f["roles"] = st.multiselect("Roles", options=opts, default=f.get("roles", []))
            if "Client Region" in df_raw.columns:
                opts = sorted([x for x in df_raw["Client Region"].dropna().unique().tolist()])
                f["regions"] = st.multiselect("Regions", options=opts, default=f.get("regions", []))
            if "Project Type" in df_raw.columns:
                opts = sorted([x for x in df_raw["Project Type"].dropna().unique().tolist()])
                f["project_types"] = st.multiselect("Project types", options=opts, default=f.get("project_types", []))

    df = apply_filters(df_raw, st.session_state.filters)

    # KPI cards
    k1, k2, k3, k4, k5 = st.columns(5)
    total_rev = float(df["Revenue"].sum()) if "Revenue" in df.columns else 0.0
    total_gm = float(df["Gross Margin"].sum()) if "Gross Margin" in df.columns else 0.0
    margin_pct = (total_gm / total_rev * 100.0) if total_rev else 0.0
    bill_hours = float(df["Billable Hours"].sum()) if "Billable Hours" in df.columns else 0.0
    sat_col = "Client Satisfaction Score" if "Client Satisfaction Score" in df.columns else None
    avg_sat = float(df[sat_col].mean()) if sat_col else 0.0

    k1.metric("Revenue", f"${total_rev:,.0f}")
    k2.metric("Gross Margin", f"${total_gm:,.0f}")
    k3.metric("Margin %", f"{margin_pct:,.1f}%")
    k4.metric("Billable Hours", f"{bill_hours:,.0f}")
    k5.metric("Avg Satisfaction", f"{avg_sat:,.2f}" if sat_col else "—")

    dash_menu = st.radio(
        "Menu",
        ["Overview", "Consultants", "Projects", "Clients"],
        horizontal=True,
        label_visibility="visible",
    )

    st.divider()

    if dash_menu == "Overview":
        c_left, c_right = st.columns([0.62, 0.38], gap="large")

        with c_left:
            fig = chart_revenue_trend(df)
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)

            cA, cB = st.columns(2)
            with cA:
                fig = chart_top_roles(df, top_n=10)
                if fig is not None:
                    st.plotly_chart(fig, use_container_width=True)
            with cB:
                fig = chart_top_clients(df, top_n=10)
                if fig is not None:
                    st.plotly_chart(fig, use_container_width=True)

        with c_right:
            if DEFAULT_SIDE_ART and os.path.exists(DEFAULT_SIDE_ART):
                ext = os.path.splitext(DEFAULT_SIDE_ART)[1].lower().lstrip(".") or "png"
                b64 = _img_b64(DEFAULT_SIDE_ART)
                st.markdown(
                    f"""
<div class="steeves-sidecard">
  <img src="data:image/{ext};base64,{b64}" alt="Project screenshot" />
</div>
""",
                    unsafe_allow_html=True,
                )

            fig = chart_top_projects(df, top_n=10)
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)

            fig = chart_margin_vs_satisfaction(df)
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)

    elif dash_menu == "Consultants":
        c1, c2 = st.columns([0.62, 0.38], gap="large")
        with c1:
            fig = chart_role_profitability(df, top_n=15)
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)
            fig = chart_utilization_by_role(df, top_n=15)
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)

        with c2:
            fig = chart_top_consultants(df, top_n=10)
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)
            fig = chart_utilization_by_location(df, top_n=10)
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)

    elif dash_menu == "Projects":
        c1, c2 = st.columns([0.62, 0.38], gap="large")
        with c1:
            fig = chart_project_timeline(df, top_n=15)
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)
            fig = chart_top_projects_by_gross_margin(df, top_n=10)
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)

        with c2:
            fig = chart_project_type_compare(df)
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)

    elif dash_menu == "Clients":
        c1, c2 = st.columns([0.62, 0.38], gap="large")
        with c1:
            fig = chart_margin_vs_satisfaction(df)
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = chart_top_clients(df, top_n=10)
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)

# ---------------- Chat ----------------
with tabs[1]:
    top_l, top_r = st.columns([0.8, 0.2], vertical_alignment="center")
    with top_l:
        st.subheader("Chat")
    with top_r:
        if st.button("Clear chat", type="secondary"):
            current = _current_session()
            current["messages"] = []
            save_session(current)
            # reset input/voice state
            st.session_state.draft_prompt = ""
            st.session_state.pending_prompt = ""
            st.session_state.last_voice_hash = ""
            st.session_state.voice_widget_nonce = (st.session_state.get("voice_widget_nonce", 0) + 1)
            st.rerun()
    current = _current_session()

    # Render messages
    for m in current.get("messages", []):
        with st.chat_message(m["role"]):
            if m.get("type") == "plot":
                st.markdown(m["content"])
                try:
                    fig = pio.from_json(m["fig_json"])
                    st.plotly_chart(fig, use_container_width=True)
                except Exception:
                    st.info("(Could not render chart from stored JSON.)")
            else:
                st.markdown(m["content"])

    # Professional chat bar with record + input + send
    st.markdown(
        """
        <style>
        .steeves-chatbar {
          background: rgba(255,255,255,0.70);
          border: 1px solid rgba(0,0,0,0.06);
          border-radius: 18px;
          padding: 10px 12px;
          box-shadow: 0 18px 60px rgba(0,0,0,0.08);
          backdrop-filter: blur(14px);
          -webkit-backdrop-filter: blur(14px);
        }
        .steeves-chatbar [data-testid="stTextInput"] input {
          border-radius: 14px !important;
        }
        .steeves-chatbar button {
          border-radius: 14px !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.session_state.setdefault("draft_prompt", "")
    st.session_state.setdefault("pending_prompt", "")
    st.session_state.setdefault("last_voice_hash", "")
    st.session_state.setdefault("voice_widget_nonce", 0)

    def _queue_send_from_enter():
        txt = (st.session_state.get("draft_prompt") or "").strip()
        if txt:
            st.session_state.pending_prompt = txt
            st.rerun()

    # Process pending prompt BEFORE creating the input widget.
    # (Streamlit doesn't allow modifying a widget's session_state key after it's instantiated.)
    if st.session_state.get("pending_prompt"):
        prompt = st.session_state.pending_prompt
        st.session_state.pending_prompt = ""
        st.session_state.draft_prompt = ""
        # Reset voice recorder widget so it's ready
        st.session_state.voice_widget_nonce += 1
        st.session_state.last_voice_hash = ""
        handle_prompt(prompt, current)

    # Voice recording: prefer Streamlit's native audio recorder (toggle + indicator).
    # This is more reliable/professional than a custom record component.
    with st.container():
        st.markdown('<div class="steeves-chatbar">', unsafe_allow_html=True)
        c_rec, c_input, c_send = st.columns([0.10, 0.75, 0.15], vertical_alignment="center")

        with c_rec:
            audio_bytes = None
            if WhisperModel is None:
                st.button("⏺", disabled=True, help="Install voice dependencies to enable transcription.")
            else:
                if hasattr(st, "audio_input"):
                    audio_file = st.audio_input(
                        " ",
                        key=f"voice_audio_{st.session_state.voice_widget_nonce}",
                        label_visibility="collapsed",
                    )
                    if audio_file is not None:
                        try:
                            audio_bytes = audio_file.getvalue()
                        except Exception:
                            audio_bytes = None
                else:
                    st.button("⏺", disabled=True, help="Update Streamlit to use built-in audio recording.")

        # If a new recording arrives, transcribe and populate the draft box
        if audio_bytes:
            h = str(hash(audio_bytes))
            if h != st.session_state.last_voice_hash:
                st.session_state.last_voice_hash = h
                with st.spinner("Transcribing…"):
                    try:
                        # Preserve the recorded file extension when possible (webm/wav/mp4)
                        suffix = ".wav"
                        try:
                            name = getattr(audio_file, "name", "") or ""
                            ext = os.path.splitext(name)[1].lower()
                            if ext in {".wav", ".webm", ".mp3", ".mp4", ".m4a", ".ogg"}:
                                suffix = ext
                        except Exception:
                            pass

                        transcript = transcribe_audio_bytes(
                            audio_bytes,
                            suffix=suffix,
                            model_size=st.session_state.settings.get("voice_model", "base"),
                        )
                    except Exception:
                        transcript = ""
                if transcript:
                    st.session_state.draft_prompt = transcript

        with c_input:
            st.text_input(
                "Message",
                key="draft_prompt",
                placeholder="Message…",
                label_visibility="collapsed",
                on_change=_queue_send_from_enter,
            )

        with c_send:
            if st.button("Send", type="primary"):
                st.session_state.pending_prompt = (st.session_state.get("draft_prompt") or "").strip()
                st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

# ---------------- Data ----------------
with tabs[2]:
    st.subheader("Data")
    df = apply_filters(df_raw, st.session_state.filters)
    st.caption(f"Rows: {len(df):,} • Columns: {df.shape[1]:,}")
    st.dataframe(df.head(200), use_container_width=True, height=420)
    st.download_button("Download filtered CSV", data=df.to_csv(index=False).encode("utf-8"), file_name="filtered_data.csv", mime="text/csv")
