from __future__ import annotations

import io
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

try:
    import plotly.express as px
    import plotly.graph_objects as go
    _PLOTLY_AVAILABLE = True
    _plotly_available = True
except Exception:
    _PLOTLY_AVAILABLE = False
    _plotly_available = False
    px = None  # type: ignore
    go = None  # type: ignore

# Bleach-inspired dark Plotly theme + orange reiryoku scale shared by every chart.
if _PLOTLY_AVAILABLE:
    _KANAME_TEMPLATE = go.layout.Template(
        layout=go.Layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#f2f2f5", family="Segoe UI, Hiragino Sans, sans-serif", size=13),
            title=dict(font=dict(color="#f2f2f5", size=16)),
            legend=dict(font=dict(color="#f2f2f5"), bgcolor="rgba(0,0,0,0)"),
            xaxis=dict(gridcolor="rgba(255,255,255,0.07)", zerolinecolor="rgba(255,255,255,0.12)",
                       linecolor="rgba(255,255,255,0.18)", tickfont=dict(color="#c9c9d4"),
                       title=dict(font=dict(color="#c9c9d4"))),
            yaxis=dict(gridcolor="rgba(255,255,255,0.07)", zerolinecolor="rgba(255,255,255,0.12)",
                       linecolor="rgba(255,255,255,0.18)", tickfont=dict(color="#c9c9d4"),
                       title=dict(font=dict(color="#c9c9d4"))),
            coloraxis=dict(colorbar=dict(outlinewidth=0, tickfont=dict(color="#c9c9d4"))),
        )
    )
    _KANAME_SCALE = [[0.0, "#1b1b24"], [0.35, "#7a2f00"], [0.7, "#cc4d00"], [1.0, "#ff8c1a"]]
else:
    _KANAME_TEMPLATE = "plotly_dark"  # type: ignore
    _KANAME_SCALE = [[0.0, "#1b1b24"], [1.0, "#ff8c1a"]]  # type: ignore

import streamlit as st

try:
    import requests
    _REQUESTS_AVAILABLE = True
except Exception:
    _REQUESTS_AVAILABLE = False

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")
NLP_BASE = f"{BACKEND_URL}/api/nlp"
VOICE_BASE = f"{BACKEND_URL}/api/voice"
DATA_BASE = f"{BACKEND_URL}/api/data"
DOC_BASE = f"{BACKEND_URL}/api/documents"
TRANS_BASE = f"{BACKEND_URL}/api/translation"
ML_BASE = f"{BACKEND_URL}/api/ml"


def _extract_filename_from_disp(disp: str, fallback: str) -> str:
    if not disp:
        return fallback
    star = None
    for part in disp.split(";"):
        p = part.strip()
        if p.lower().startswith("filename*="):
            star = p.split("=", 1)[1].strip()
            if star.lower().startswith("utf-8''"):
                from urllib.parse import unquote
                star = unquote(star[7:])
                return star
    for part in disp.split(";"):
        p = part.strip()
        if p.lower().startswith("filename="):
            return p.split("=", 1)[1].strip().strip('"')
    return fallback

st.set_page_config(
    page_title="KANAME [要] — AI Intelligence System",
    page_icon="要",
    layout="wide",
    initial_sidebar_state="expanded",
)


DEFAULT_NER_ANNOTATIONS = """[
  {
    "text": "田中は東京へ行く。",
    "entities": [
      {"start": 0, "end": 2, "label": "PERSON"},
      {"start": 3, "end": 5, "label": "LOCATION"}
    ]
  },
  {
    "text": "佐藤は大阪へ行く。",
    "entities": [
      {"start": 0, "end": 2, "label": "PERSON"},
      {"start": 3, "end": 5, "label": "LOCATION"}
    ]
  }
]"""


CUSTOM_CSS = """
<style>
    /* Typefaces resolve to locally-installed system fonts (no external CDN
       request, so no blocked/aborted font fetch and no offline dependency). */
    /* ===================================================================
       KANAME [要] — "SOUL REAPER" dark theme (Bleach-inspired)
       Black canvas + white contrast + orange reiryoku accent.
       =================================================================== */
    :root {
        --kaname-void: #0a0a0d;
        --kaname-ink: #050507;
        --kaname-panel: #15151c;
        --kaname-panel-2: #1b1b24;
        --kaname-line: #2a2a36;
        --kaname-white: #f2f2f5;
        --kaname-muted: #9a9aa8;
        --kaname-orange: #ff6a00;
        --kaname-orange-bright: #ff8c1a;
        --kaname-orange-deep: #cc4d00;
        --kaname-blood: #c1121f;
        /* Display (latin): condensed technical sans -> Bahnschrift on Windows. */
        --kaname-font-tech: 'Bahnschrift SemiCondensed', 'Bahnschrift',
            'Segoe UI Variable Display', 'Segoe UI', 'Helvetica Neue', Arial, sans-serif;
        /* Display (japanese): mincho serif -> Yu Mincho on Windows. */
        --kaname-font-jp: 'Yu Mincho', 'YuMincho', 'Yu Mincho Demibold',
            'Hiragino Mincho ProN', 'Noto Serif JP', 'Times New Roman', serif;
    }

    /* ---------- Global canvas ---------- */
    .stApp {
        background:
            radial-gradient(1200px 600px at 15% -10%, rgba(255,106,0,0.10), transparent 60%),
            radial-gradient(900px 500px at 110% 10%, rgba(255,106,0,0.05), transparent 55%),
            var(--kaname-void);
        color: var(--kaname-white);
    }
    .main { background-color: transparent; }
    .block-container { padding-top: 2.2rem; max-width: 1400px; }

    /* Faint kanji watermark behind the whole app */
    .stApp::before {
        content: "死神";
        position: fixed;
        top: 40%; left: 50%;
        transform: translate(-50%, -50%);
        font-size: 42rem; font-weight: 900;
        color: rgba(255,106,0,0.022);
        z-index: 0; pointer-events: none;
        letter-spacing: 0.05em;
        user-select: none;
    }
    .stApp::after {
        content: "要";
        position: fixed;
        bottom: 2%; right: 3%;
        font-size: 16rem; font-weight: 900;
        color: rgba(255,255,255,0.02);
        z-index: 0; pointer-events: none;
        user-select: none;
    }
    section.main > div { position: relative; z-index: 1; }

    /* ---------- Typography ---------- */
    h1, h2, h3, h4, h5, h6,
    [data-testid="stMarkdownContainer"] h1,
    [data-testid="stMarkdownContainer"] h2,
    [data-testid="stMarkdownContainer"] h3 {
        font-family: 'Segoe UI', 'Hiragino Sans', 'Yu Gothic', sans-serif;
        color: var(--kaname-white) !important;
        letter-spacing: 0.5px;
    }
    h1 { border-bottom: 2px solid rgba(255,106,0,0.35); padding-bottom: 0.35rem; }
    h2 { border-left: 4px solid var(--kaname-orange); padding-left: 0.6rem; }
    p, span, label, li, .stMarkdown { color: var(--kaname-white); }
    a { color: var(--kaname-orange-bright) !important; }
    hr { border-color: var(--kaname-line) !important; }

    /* Main-area reading text: slightly larger for comfort (matches the sidebar) */
    [data-testid="stAppViewContainer"] { font-size: 1.06rem; }
    [data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] li {
        font-size: 1.06rem; line-height: 1.7;
    }

    /* ---------- Hero / branding ---------- */
    .kaname-title {
        font-size: 2.9rem; font-weight: 900; color: var(--kaname-white);
        letter-spacing: 3px; margin-bottom: 0;
        text-shadow: 0 0 18px rgba(255,106,0,0.35), 0 0 42px rgba(255,106,0,0.18);
    }
    .kaname-subtitle {
        font-size: 1.1rem; color: var(--kaname-muted); font-weight: 400; margin-top: 0.25rem;
        letter-spacing: 0.5px;
    }
    .kaname-kanji {
        font-size: 4.8rem; color: var(--kaname-orange); font-weight: 900; line-height: 1;
        text-shadow: 0 0 22px rgba(255,106,0,0.75), 0 0 60px rgba(255,106,0,0.35);
        animation: reiatsu-pulse 3.2s ease-in-out infinite;
        display: inline-block;
    }
    @keyframes reiatsu-pulse {
        0%, 100% { text-shadow: 0 0 18px rgba(255,106,0,0.55), 0 0 44px rgba(255,106,0,0.25); transform: scale(1); }
        50%      { text-shadow: 0 0 30px rgba(255,140,26,0.95), 0 0 80px rgba(255,106,0,0.55); transform: scale(1.04); }
    }
    .kaname-hero-fade { animation: ink-rise 0.7s ease-out both; }
    @keyframes ink-rise {
        from { opacity: 0; transform: translateY(10px); filter: blur(3px); }
        to   { opacity: 1; transform: translateY(0);    filter: blur(0); }
    }

    /* ---------- Ink / brush-stroke divider ---------- */
    .divider, hr.divider {
        border: none;
        height: 10px;
        margin: 1.4rem 0;
        background:
            linear-gradient(to right,
                transparent 0%,
                rgba(255,106,0,0.10) 6%,
                rgba(255,106,0,0.85) 22%,
                rgba(255,140,26,0.95) 34%,
                rgba(255,106,0,0.35) 44%,
                rgba(242,242,245,0.16) 62%,
                rgba(255,106,0,0.45) 80%,
                transparent 100%);
        box-shadow: 0 0 14px rgba(255,106,0,0.28);
        -webkit-mask-image: radial-gradient(ellipse 60% 100% at 50% 50%, #000 55%, transparent 100%);
        mask-image: radial-gradient(ellipse 60% 100% at 50% 50%, #000 55%, transparent 100%);
    }

    /* ---------- Cards ---------- */
    .module-card {
        background: linear-gradient(160deg, var(--kaname-panel-2), var(--kaname-panel));
        border: 1px solid var(--kaname-line);
        border-top: 3px solid var(--kaname-orange);
        border-radius: 12px;
        padding: 1.25rem;
        box-shadow: 0 6px 22px rgba(0,0,0,0.45);
        transition: box-shadow 0.25s ease, transform 0.25s ease, border-color 0.25s ease;
        height: 100%;
    }
    .module-card:hover {
        box-shadow: 0 10px 34px rgba(255,106,0,0.30), 0 0 0 1px rgba(255,106,0,0.25);
        transform: translateY(-4px);
        border-top-color: var(--kaname-orange-bright);
    }
    .status-badge {
        display: inline-block; padding: 4px 12px; border-radius: 20px;
        font-size: 0.75rem; font-weight: 700; letter-spacing: 0.4px;
        text-transform: uppercase;
    }
    .status-foundation { background: rgba(255,106,0,0.10); color: #ffb877; border: 1px solid rgba(255,106,0,0.35); }
    .status-impl       { background: rgba(34,197,94,0.12);  color: #6ee7a0; border: 1px solid rgba(34,197,94,0.35); }
    .status-warn       { background: rgba(255,106,0,0.18);  color: #ff9a4d; border: 1px solid rgba(255,106,0,0.5); }

    /* ---------- KPI ---------- */
    .kpi-card {
        background: linear-gradient(160deg, var(--kaname-panel-2), var(--kaname-panel));
        border-radius: 10px; padding: 1rem 1.25rem;
        border: 1px solid var(--kaname-line);
        border-left: 3px solid var(--kaname-orange);
        box-shadow: 0 4px 16px rgba(0,0,0,0.4);
    }
    .kpi-label { color: var(--kaname-muted); font-size: 0.8rem; font-weight: 600; letter-spacing: 0.5px; text-transform: uppercase; }
    .kpi-value { color: var(--kaname-white); font-size: 1.75rem; font-weight: 800; margin-top: 0.15rem; text-shadow: 0 0 16px rgba(255,106,0,0.25); }
    .kpi-delta-up { color: #6ee7a0; font-size: 0.8rem; }
    .kpi-delta-down { color: #ff6b6b; font-size: 0.8rem; }

    /* ---------- Backend indicator ---------- */
    .backend-indicator {
        display: inline-flex; align-items: center; gap: 8px;
        padding: 6px 14px; border-radius: 999px;
        font-size: 0.85rem; font-weight: 700; letter-spacing: 0.3px;
    }
    .bi-online  { background: rgba(34,197,94,0.12); color: #6ee7a0; border: 1px solid rgba(34,197,94,0.4); box-shadow: 0 0 14px rgba(34,197,94,0.22); }
    .bi-offline { background: rgba(193,18,31,0.16); color: #ff8a8a; border: 1px solid rgba(193,18,31,0.5); box-shadow: 0 0 14px rgba(193,18,31,0.25); }

    /* ---------- Sidebar (captain's black) ---------- */
    [data-testid="stSidebar"] {
        background:
            radial-gradient(620px 320px at 50% -6%, rgba(255,106,0,0.10), transparent 62%),
            linear-gradient(180deg, #0d0d12, #050507) !important;
        border-right: 1px solid rgba(255,106,0,0.25);
        box-shadow: 6px 0 34px rgba(0,0,0,0.6);
    }
    [data-testid="stSidebar"] > div:first-child { padding-top: 1.1rem; }
    [data-testid="stSidebar"]::before {
        content: "卍";
        position: absolute; bottom: 8px; right: 12px;
        font-size: 7rem; font-weight: 900;
        color: rgba(255,106,0,0.045);
        pointer-events: none;
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label { color: var(--kaname-white) !important; }

    /* Brand block */
    .soul-brand { display: flex; align-items: center; gap: 12px; padding: 0.2rem 0.1rem; }
    .soul-brand-mark {
        width: 52px; height: 52px; flex: 0 0 52px;
        display: flex; align-items: center; justify-content: center;
        font-family: var(--kaname-font-jp); font-weight: 900; font-size: 1.8rem;
        color: var(--kaname-orange);
        border: 1.5px solid rgba(255,106,0,0.55); border-radius: 12px;
        background: rgba(255,106,0,0.08);
        text-shadow: 0 0 16px rgba(255,106,0,0.85);
        box-shadow: 0 0 18px rgba(255,106,0,0.25), inset 0 0 12px rgba(255,106,0,0.12);
        animation: reiatsu-pulse 3.4s ease-in-out infinite;
    }
    .soul-brand-name {
        font-family: var(--kaname-font-tech); font-weight: 700;
        font-size: 1.5rem; letter-spacing: 4px; line-height: 1;
        color: var(--kaname-white);
        text-shadow: 0 0 14px rgba(255,106,0,0.4);
    }
    .soul-brand-sub {
        font-family: var(--kaname-font-tech); font-size: 0.6rem; font-weight: 600;
        letter-spacing: 3px; color: var(--kaname-orange) !important;
        margin-top: 4px; text-transform: uppercase;
    }
    .soul-divider {
        border: none; height: 2px; margin: 0.9rem 0 0.7rem;
        background: linear-gradient(90deg, rgba(255,106,0,0.9), rgba(255,140,26,0.25) 55%, transparent);
        box-shadow: 0 0 10px rgba(255,106,0,0.35); border-radius: 2px;
    }
    .sidebar-section {
        display: flex; align-items: center; gap: 8px;
        font-family: var(--kaname-font-tech); font-weight: 700 !important;
        font-size: 0.72rem; letter-spacing: 3px; text-transform: uppercase;
        color: var(--kaname-muted) !important;
        margin: 0.55rem 0 0.5rem 3px;
    }
    .sidebar-section::before {
        content: ""; width: 4px; height: 14px; border-radius: 2px;
        background: var(--kaname-orange); box-shadow: 0 0 10px rgba(255,106,0,0.75);
    }

    /* Navigation radio -> nav items */
    [data-testid="stSidebar"] [role="radiogroup"] { gap: 5px; display: flex; flex-direction: column; }
    [data-testid="stSidebar"] [role="radiogroup"] label {
        display: flex; align-items: center; gap: 11px;
        width: 100%; padding: 10px 14px; margin: 0 16px 0 0;
        border-radius: 10px; border-left: 3px solid transparent;
        background: rgba(255,255,255,0.015);
        transition: background .18s ease, border-color .18s ease, transform .18s ease,
                    margin-right .24s ease, box-shadow .24s ease;
        cursor: pointer;
    }
    [data-testid="stSidebar"] [role="radiogroup"] label p {
        font-family: var(--kaname-font-tech);
        font-weight: 600; font-size: 1.13rem; letter-spacing: 0.5px; margin: 0;
    }
    [data-testid="stSidebar"] [role="radiogroup"] label:hover {
        background: rgba(255,106,0,0.12);
        border-left-color: rgba(255,106,0,0.6);
        transform: translateX(3px);
    }
    [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
        margin-right: 0;
        background: linear-gradient(90deg, rgba(255,106,0,0.24), rgba(255,106,0,0.02));
        border-left-color: var(--kaname-orange);
        box-shadow: 0 0 18px rgba(255,106,0,0.25), inset 0 0 18px rgba(255,106,0,0.06);
    }
    [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) p {
        color: var(--kaname-orange-bright) !important;
        text-shadow: 0 0 12px rgba(255,106,0,0.5);
    }
    [data-testid="stSidebar"] [role="radiogroup"] label:hover p { color: #ffc880 !important; }

    /* Sidebar footer */
    .soul-footer {
        margin-top: 0.7rem; padding: 0.6rem 0.7rem;
        border: 1px solid var(--kaname-line); border-radius: 10px;
        background: rgba(255,255,255,0.02);
        font-family: var(--kaname-font-tech); font-size: 0.68rem; font-weight: 600;
        letter-spacing: 2px; color: var(--kaname-muted) !important;
        text-transform: uppercase; text-align: center; line-height: 1.8;
    }
    .soul-footer b { color: var(--kaname-orange) !important; }
    [data-testid="stSidebar"] .backend-indicator { width: 100%; justify-content: center; }

    /* ---------- Inputs / selects / textareas ---------- */
    .stTextInput input, .stNumberInput input, .stTextArea textarea,
    [data-baseweb="input"] input, [data-baseweb="textarea"] textarea,
    div[data-baseweb="select"] > div {
        background-color: var(--kaname-panel-2) !important;
        color: var(--kaname-white) !important;
        border: 1px solid var(--kaname-line) !important;
        border-radius: 8px !important;
    }
    .stTextInput input:focus, .stNumberInput input:focus, .stTextArea textarea:focus,
    div[data-baseweb="select"] > div:focus-within {
        border-color: var(--kaname-orange) !important;
        box-shadow: 0 0 0 2px rgba(255,106,0,0.30) !important;
    }
    [data-baseweb="popover"] div, [role="listbox"], [role="option"] {
        background-color: var(--kaname-panel-2) !important;
        color: var(--kaname-white) !important;
    }
    [role="option"]:hover { background-color: rgba(255,106,0,0.18) !important; }
    .stTextInput label, .stTextArea label, .stNumberInput label,
    .stSelectbox label, .stMultiSelect label, .stSlider label,
    .stRadio label, .stCheckbox label, .stFileUploader label,
    [data-testid="stWidgetLabel"] { color: var(--kaname-white) !important; }

    /* ---------- Buttons ---------- */
    .stButton > button, .stDownloadButton > button, .stFormSubmitButton > button {
        background: linear-gradient(160deg, var(--kaname-panel-2), var(--kaname-panel));
        color: var(--kaname-white) !important;
        border: 1px solid var(--kaname-orange-deep);
        border-radius: 8px; font-weight: 700; letter-spacing: 0.4px;
        transition: all 0.2s ease;
    }
    .stButton > button:hover, .stDownloadButton > button:hover, .stFormSubmitButton > button:hover {
        border-color: var(--kaname-orange-bright);
        color: var(--kaname-orange-bright) !important;
        box-shadow: 0 0 18px rgba(255,106,0,0.40);
    }
    .stButton > button:active, .stDownloadButton > button:active { transform: translateY(1px); }

    /* ---------- Tabs ---------- */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px; background: transparent;
        border-bottom: 1px solid var(--kaname-line);
    }
    .stTabs [data-baseweb="tab"] {
        background: var(--kaname-panel);
        color: var(--kaname-muted) !important;
        border: 1px solid var(--kaname-line);
        border-radius: 8px 8px 0 0;
        font-weight: 700;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(180deg, rgba(255,106,0,0.22), var(--kaname-panel)) !important;
        color: var(--kaname-orange-bright) !important;
        border-bottom: 2px solid var(--kaname-orange) !important;
        box-shadow: 0 -3px 16px rgba(255,106,0,0.30);
    }

    /* ---------- Expander ---------- */
    [data-testid="stExpander"] {
        background: var(--kaname-panel);
        border: 1px solid var(--kaname-line) !important;
        border-radius: 10px !important;
    }
    [data-testid="stExpander"] summary, [data-testid="stExpander"] summary * {
        color: var(--kaname-white) !important; font-weight: 600;
    }
    [data-testid="stExpander"] summary:hover { color: var(--kaname-orange-bright) !important; }

    /* ---------- Dataframe / table ---------- */
    [data-testid="stDataFrame"], [data-testid="stTable"] {
        border: 1px solid var(--kaname-line); border-radius: 10px; overflow: hidden;
    }
    [data-testid="stDataFrame"] * , [data-testid="stTable"] * { color: var(--kaname-white) !important; }
    [data-testid="stDataFrame"] canvas { background: transparent !important; }

    /* ---------- Metrics ---------- */
    [data-testid="stMetric"] {
        background: linear-gradient(160deg, var(--kaname-panel-2), var(--kaname-panel));
        border: 1px solid var(--kaname-line); border-left: 3px solid var(--kaname-orange);
        border-radius: 10px; padding: 0.8rem 1rem;
    }
    [data-testid="stMetricLabel"], [data-testid="stMetricLabel"] * { color: var(--kaname-muted) !important; }
    [data-testid="stMetricValue"], [data-testid="stMetricValue"] * { color: var(--kaname-white) !important; }

    /* ---------- Alerts / info / success / warning ---------- */
    [data-testid="stAlert"] { border-radius: 10px; border: 1px solid var(--kaname-line); }

    /* ---------- Progress / slider ---------- */
    .stProgress > div > div > div { background-image: linear-gradient(90deg, var(--kaname-orange-deep), var(--kaname-orange-bright)); }
    [data-baseweb="slider"] div[role="slider"] { background-color: var(--kaname-orange) !important; }

    /* ---------- Code / JSON ---------- */
    code, pre, .stCodeBlock { background: var(--kaname-ink) !important; color: #ffb877 !important; border-radius: 6px; }

    /* ---------- File uploader ---------- */
    [data-testid="stFileUploaderDropzone"] {
        background: var(--kaname-panel); border: 1px dashed var(--kaname-orange-deep); border-radius: 10px;
    }

    /* ---------- Scrollbar ---------- */
    ::-webkit-scrollbar { width: 10px; height: 10px; }
    ::-webkit-scrollbar-track { background: var(--kaname-ink); }
    ::-webkit-scrollbar-thumb { background: #33333f; border-radius: 6px; }
    ::-webkit-scrollbar-thumb:hover { background: var(--kaname-orange-deep); }

    /* ---------- Toast / spinner accent ---------- */
    .stSpinner > div { border-top-color: var(--kaname-orange) !important; }

    /* ===================================================================
       IMMERSIVE LAYER — fonts, reiatsu background, cinematic hero, motion
       =================================================================== */

    /* ---------- Typography upgrade ---------- */
    h1, h2, h3 { font-family: var(--kaname-font-tech) !important; letter-spacing: 1px; }
    h1 { font-weight: 700; }
    .kaname-latin { font-family: var(--kaname-font-tech); }

    /* ---------- Reiatsu particle field (live, CSS-only) ---------- */
    .reiatsu-layer {
        position: fixed; inset: 0; z-index: 0;
        pointer-events: none; overflow: hidden;
    }
    .reiatsu-particle {
        position: absolute; bottom: -30px; border-radius: 50%;
        background: radial-gradient(circle, #ffc880 0%, #ff6a00 45%, rgba(255,106,0,0) 72%);
        box-shadow: 0 0 14px rgba(255,106,0,0.75);
        animation-name: reiatsu-rise;
        animation-timing-function: linear;
        animation-iteration-count: infinite;
        will-change: transform, opacity;
    }
    @keyframes reiatsu-rise {
        0%   { transform: translateY(0) translateX(0) scale(1);   opacity: 0; }
        12%  { opacity: 1; }
        50%  { transform: translateY(-52vh) translateX(26px) scale(1.35); }
        88%  { opacity: 0.9; }
        100% { transform: translateY(-106vh) translateX(-18px) scale(0.7); opacity: 0; }
    }
    /* Slow drifting aura blobs behind everything */
    .reiatsu-aura {
        position: fixed; z-index: 0; pointer-events: none; border-radius: 50%;
        filter: blur(70px); opacity: 0.5;
    }
    .aura-1 { width: 46vw; height: 46vw; left: -12vw; top: -10vw;
              background: radial-gradient(circle, rgba(255,106,0,0.28), transparent 65%);
              animation: aura-drift 26s ease-in-out infinite alternate; }
    .aura-2 { width: 40vw; height: 40vw; right: -10vw; bottom: -14vw;
              background: radial-gradient(circle, rgba(255,140,26,0.20), transparent 65%);
              animation: aura-drift 34s ease-in-out infinite alternate-reverse; }
    @keyframes aura-drift {
        from { transform: translate(0, 0) scale(1); }
        to   { transform: translate(6vw, 5vh) scale(1.25); }
    }

    /* ---------- Cinematic hero banner ---------- */
    .hero-banner {
        position: relative; overflow: hidden;
        border-radius: 18px; min-height: 360px;
        border: 1px solid rgba(255,106,0,0.40);
        box-shadow: 0 26px 70px rgba(0,0,0,0.65), inset 0 0 60px rgba(255,106,0,0.06);
        background-color: #08080b;
        animation: hero-in 0.9s cubic-bezier(.2,.7,.2,1) both;
    }
    @keyframes hero-in { from { opacity: 0; transform: translateY(24px) scale(0.985); } to { opacity: 1; transform: none; } }
    .hero-art {
        position: absolute; inset: -4%;
        background-size: cover; background-position: center 30%;
        background-repeat: no-repeat;
        animation: hero-kenburns 22s ease-in-out infinite alternate;
        will-change: transform;
    }
    @keyframes hero-kenburns { from { transform: scale(1.04) translateY(0); } to { transform: scale(1.16) translateY(-2%); } }
    .hero-scrim {
        position: absolute; inset: 0;
        background:
            linear-gradient(90deg, rgba(5,5,8,0.94) 0%, rgba(5,5,8,0.72) 42%, rgba(5,5,8,0.25) 100%),
            linear-gradient(0deg, rgba(5,5,8,0.9) 0%, rgba(5,5,8,0.1) 55%);
    }
    .hero-glow {
        position: absolute; inset: 0; pointer-events: none;
        background: radial-gradient(700px 320px at 18% 88%, rgba(255,106,0,0.42), transparent 62%);
        mix-blend-mode: screen;
        animation: reiatsu-pulse 4.5s ease-in-out infinite;
    }
    .hero-content { position: relative; z-index: 3; padding: 2.6rem 2.8rem; max-width: 900px; }
    .hero-kanji {
        font-family: var(--kaname-font-jp); font-weight: 900;
        font-size: 5.6rem; line-height: 0.9; color: var(--kaname-orange);
        text-shadow: 0 0 24px rgba(255,106,0,0.85), 0 0 70px rgba(255,106,0,0.45);
        animation: reiatsu-pulse 3.4s ease-in-out infinite;
        display: inline-block;
    }
    .hero-title {
        font-family: var(--kaname-font-tech); font-weight: 700;
        font-size: 4.1rem; letter-spacing: 10px; line-height: 1;
        margin: 0.2rem 0 0.3rem; border: none !important; padding: 0;
        color: var(--kaname-white);
        text-shadow: 0 0 22px rgba(255,106,0,0.45), 0 0 60px rgba(255,106,0,0.2);
    }
    .hero-sub {
        font-family: var(--kaname-font-jp);
        font-size: 1.05rem; color: #d7d7e0; letter-spacing: 0.6px;
        margin-bottom: 1rem;
    }
    .hero-tags { margin: 0.2rem 0 1rem; }
    .hero-tag {
        display: inline-block; margin: 0 6px 6px 0; padding: 4px 12px;
        font-family: var(--kaname-font-tech); font-weight: 600; letter-spacing: 1px;
        font-size: 0.78rem; text-transform: uppercase;
        color: #ffc880; border: 1px solid rgba(255,106,0,0.45);
        border-radius: 999px; background: rgba(255,106,0,0.08);
    }
    /* Hanko seal stamp */
    .hero-seal {
        position: absolute; right: 26px; bottom: 22px; z-index: 3;
        width: 74px; height: 74px; display: flex; align-items: center; justify-content: center;
        font-family: var(--kaname-font-jp); font-weight: 900; font-size: 2.3rem;
        color: #fff5ec; background: linear-gradient(160deg, #e03a10, #a50f0f);
        border-radius: 10px; transform: rotate(-7deg);
        box-shadow: 0 0 22px rgba(255,60,20,0.55), inset 0 0 12px rgba(0,0,0,0.35);
        border: 2px solid rgba(255,200,160,0.6);
        animation: seal-in 1.2s ease-out 0.5s both;
    }
    @keyframes seal-in { from { opacity: 0; transform: rotate(-7deg) scale(2.4); } to { opacity: 1; transform: rotate(-7deg) scale(1); } }

    /* ---------- Animated card frame + entrance motion ---------- */
    .module-card { animation: card-rise 0.55s ease-out backwards; }
    .module-card::after {
        content: ""; position: absolute; inset: 0; border-radius: 12px; padding: 1px;
        background: linear-gradient(120deg, rgba(255,106,0,0) 0%, rgba(255,140,26,0.9) 50%, rgba(255,106,0,0) 100%);
        background-size: 220% 100%;
        -webkit-mask: linear-gradient(#000 0 0) content-box, linear-gradient(#000 0 0);
        -webkit-mask-composite: xor; mask-composite: exclude;
        opacity: 0; transition: opacity 0.3s ease; pointer-events: none;
    }
    .module-card:hover::after { opacity: 1; animation: border-flow 2.6s linear infinite; }
    @keyframes border-flow { from { background-position: 220% 0; } to { background-position: -220% 0; } }
    @keyframes card-rise { from { opacity: 0; transform: translateY(18px); } to { opacity: 1; transform: translateY(0); } }
    .kpi-card { animation: card-rise 0.5s ease-out backwards; }

    /* ---------- Animated brush divider ---------- */
    hr.divider { animation: divider-sweep 1.1s ease-out both; }
    @keyframes divider-sweep {
        from { opacity: 0; transform: scaleX(0.2); filter: blur(6px); }
        to   { opacity: 1; transform: scaleX(1);   filter: blur(0); }
    }

    /* ---------- Section heading accent ---------- */
    h3 { position: relative; }
    h3::after {
        content: ""; display: block; width: 54px; height: 3px; margin-top: 6px;
        background: linear-gradient(90deg, var(--kaname-orange), transparent);
        border-radius: 2px;
    }
    .module-card h3::after { display: none; }
    .block-container { animation: page-in 0.5s ease-out both; }
    @keyframes page-in { from { opacity: 0; } to { opacity: 1; } }

    @media (prefers-reduced-motion: reduce) {
        .reiatsu-particle, .reiatsu-aura, .hero-art, .hero-glow, .hero-seal, .hero-kanji, .kaname-kanji { animation: none !important; }
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Immersive "Soul Reaper" theming layer
# ---------------------------------------------------------------------------

def _img_url(prompt: str, size: str = "landscape_16_9") -> str:
    from urllib.parse import quote
    return (
        "https://coresg-normal.trae.ai/api/ide/v1/text_to_image"
        f"?prompt={quote(prompt)}&image_size={size}"
    )


HERO_ART = _img_url(
    "cinematic anime illustration, lone soul reaper in black shihakusho robe with a giant katana, "
    "glowing orange spiritual energy aura, rising ember particles, swirling black ink brush strokes, "
    "pitch black background, dramatic orange rim light, ultra detailed, wide cinematic wallpaper",
    "landscape_16_9",
)


def _reiatsu_background() -> None:
    """Fixed full-viewport layer of rising reiryoku particles + drifting aura."""
    import random
    rnd = random.Random(42)
    particles = []
    for _ in range(28):
        left = rnd.uniform(0, 100)
        size = rnd.uniform(3, 9)
        delay = rnd.uniform(0, 14)
        dur = rnd.uniform(10, 22)
        op = rnd.uniform(0.25, 0.8)
        particles.append(
            f'<span class="reiatsu-particle" style="left:{left:.1f}%;'
            f'width:{size:.1f}px;height:{size:.1f}px;'
            f'animation-delay:-{delay:.1f}s;animation-duration:{dur:.1f}s;'
            f'opacity:{op:.2f};"></span>'
        )
    st.markdown(
        '<div class="reiatsu-layer" aria-hidden="true">'
        '<span class="reiatsu-aura aura-1"></span>'
        '<span class="reiatsu-aura aura-2"></span>'
        + "".join(particles)
        + "</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Backend helpers
# ---------------------------------------------------------------------------

def _request(method: str, path: str, payload: Optional[Dict] = None,
             files: Optional[Dict] = None, data: Optional[Dict] = None,
             timeout: int = 60, raw: bool = False,
             params: Optional[Dict] = None) -> Tuple[bool, Any]:
    url = f"{BACKEND_URL}{path}" if path.startswith("/") else f"{BACKEND_URL}/{path}"
    try:
        if method.upper() == "GET":
            r = requests.get(url, params=params, timeout=timeout)
        elif method.upper() == "POST":
            if files:
                r = requests.post(url, files=files, data=data, params=params, timeout=timeout)
            elif data and not payload:
                r = requests.post(url, data=data, params=params, timeout=timeout)
            else:
                r = requests.post(url, json=payload or {}, params=params, timeout=timeout)
        else:
            return False, f"Unsupported method: {method}"
        if raw:
            return (200 <= r.status_code < 300), r
        if 200 <= r.status_code < 300:
            try:
                return True, r.json()
            except Exception:
                return True, r.content
        return False, f"HTTP {r.status_code}: {r.text[:300]}"
    except requests.exceptions.ConnectionError:
        return False, (
            f"Cannot reach KANAME backend at {BACKEND_URL}.\n"
            "Start it with: uvicorn backend.main:app --reload --port 8000"
        )
    except Exception as e:
        return False, f"Request error: {e}"


def backend_online() -> Tuple[bool, str, float]:
    if not _REQUESTS_AVAILABLE:
        return False, "requests package missing", 0.0
    import time
    t0 = time.perf_counter()
    ok, data = _request("GET", "/health", timeout=5)
    dt = time.perf_counter() - t0
    if ok and isinstance(data, dict) and data.get("status") == "KANAME backend is running":
        return True, "Online", dt
    return False, (data if isinstance(data, str) else "Offline"), dt


def fetch_history(limit: int = 50) -> Tuple[bool, Any]:
    return _request("GET", f"/api/nlp/history?limit={limit}", timeout=20)


def fetch_voice_jobs_from_history(limit: int = 30) -> Tuple[bool, Any]:
    return _request("GET", f"/api/voice/jobs?limit={limit}", timeout=20)


# ---------------------------------------------------------------------------
# MODULE: DASHBOARD
# ---------------------------------------------------------------------------

def _module_card(title: str, status_label: str, status_cls: str,
                 desc: str, future: str, accent: str = "#ff8c1a") -> str:
    return f"""
    <div class="module-card">
      <div style="display:flex;justify-content:space-between;align-items:center;">
        <h3 style="margin:0;color:{accent};">{title}</h3>
        <span class="status-badge {status_cls}">{status_label}</span>
      </div>
      <p style="color:#c9c9d4;margin-top:0.9rem;line-height:1.55;">{desc}</p>
      <p style="font-size:0.85rem;color:#9a9aa8;margin-top:0.75rem;">
        <b>Next:</b> {future}
      </p>
    </div>
    """


def _kpi(label: str, value: str, delta: Optional[str] = None, delta_up: bool = True) -> str:
    delta_html = ""
    if delta:
        cls = "kpi-delta-up" if delta_up else "kpi-delta-down"
        delta_html = f'<div class="{cls}">{delta}</div>'
    return f"""
    <div class="kpi-card">
      <div class="kpi-label">{label}</div>
      <div class="kpi-value">{value}</div>
      {delta_html}
    </div>
    """


def render_dashboard():
    online, status, latency_ms = backend_online()
    cls = "bi-online" if online else "bi-offline"
    dot = "🟢" if online else "🔴"
    lat = f" · {int(latency_ms * 1000)} ms" if online else ""
    st.markdown(
        f'<div class="hero-banner">'
        f'<div class="hero-art" style="background-image:url(\'{HERO_ART}\');"></div>'
        f'<div class="hero-scrim"></div>'
        f'<div class="hero-glow"></div>'
        f'<div class="hero-content">'
        f'<div class="hero-kanji">要</div>'
        f'<h1 class="hero-title">KANAME</h1>'
        f'<div class="hero-sub">AI-Powered Japanese Language &amp; Productivity Intelligence System</div>'
        f'<div class="hero-tags">'
        f'<span class="hero-tag">Japanese NLP</span>'
        f'<span class="hero-tag">Voice</span>'
        f'<span class="hero-tag">Translation JA&#8596;EN</span>'
        f'<span class="hero-tag">Data Intelligence</span>'
        f'<span class="hero-tag">Documents</span>'
        f'<span class="hero-tag">Machine Learning</span>'
        f'</div>'
        f'<div class="backend-indicator {cls}">{dot} Backend {status}{lat}</div>'
        f'</div>'
        f'<div class="hero-seal">要</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # ---- KPI Row (live from backend if online) ----
    total_nlp = 0
    total_tokens = 0
    total_jp_words = 0
    avg_tokens_per_input = 0

    if online:
        ok_h, hist = fetch_history(limit=1000)
        if ok_h and isinstance(hist, dict):
            items = hist.get("history", []) or []
            total_nlp = len(items)
            if total_nlp:
                total_tokens = sum(int(i.get("token_count") or 0) for i in items)
                nouns_per = [len(i.get("noun_list") or []) for i in items]
                total_jp_words = sum(nouns_per)
                avg_tokens_per_input = round(total_tokens / total_nlp, 1)

    k1, k2, k3, k4 = st.columns(4)
    k1.markdown(_kpi("Total NLP Analyses", f"{total_nlp:,}"), unsafe_allow_html=True)
    k2.markdown(_kpi("Total Tokens Processed", f"{total_tokens:,}"), unsafe_allow_html=True)
    k3.markdown(_kpi("Noun Extractions", f"{total_jp_words:,}"), unsafe_allow_html=True)
    k4.markdown(_kpi("Avg Tokens / Input", f"{avg_tokens_per_input}"), unsafe_allow_html=True)

    st.markdown("")
    st.markdown("### Project Overview")
    st.write(
        "KANAME [要] is an Artificial Intelligence system focused on Japanese language processing, "
        "voice intelligence, translation, data analysis, and bilingual document generation. "
        "This dashboard serves as the central interface for all intelligence modules."
    )

    st.write("Build by Aditya Prakash Tripathy")

    with st.expander("🏗 Architecture Overview", expanded=False):
        st.markdown(
            """
            **```
            USER → Streamlit (Main UI :8501) → FastAPI Backend (:8000) → AI / NLP Engines → SQLite
                                                                              ↳ Gradio (:7860) specialized voice AI
            ```**
            - **Streamlit:** Primary dashboard & module UIs
            - **Gradio:** Voice / microphone interaction (STT + post-NLP)
            - **FastAPI:** Backend services + module routers (nlp, voice, data, translation, documents, ml)
            - **AI Engines:** SudachiPy NLP · faster-whisper STT · pandas/Plotly data intel · python-docx/ReportLab docs
            - **SQLite:** Persistent storage (NlpAnalysis, VoiceJob, DatasetRecord)
            """
        )

    st.markdown("")
    st.markdown("### Intelligence Modules")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(_module_card(
            "Japanese NLP",
            "Implemented",
            "status-impl",
            "Tokenization, morphological analysis, POS tagging, readings, wakati-gaki, POS-based filtering, SQLite-backed history, TF-IDF keywords, rule-based sentiment, and dual named-entity recognition (Sudachi dictionary/POS spans + supervised span-trained NER with train/predict/persist APIs).",
            "Custom fine-tuned NER on larger annotated corpora, dependency parsing, semantic role labeling.",
            "#ff8c1a",
        ), unsafe_allow_html=True)
    with c2:
        st.markdown(_module_card(
            "Voice Intelligence",
            "Implemented",
            "status-impl",
            "Speech-to-Text using faster-whisper (tiny→large-v3). Audio upload & mic via Gradio. Persists VoiceJobs to DB. Post-NLP via Sudachi. Voice command parsing, Japanese TTS (optional pyttsx3), and speaker diarization (optional pyannote.audio).",
            "Streaming STT, neural TTS voices, multi-speaker turn-taking.",
            "#ff6a00",
        ), unsafe_allow_html=True)
    with c3:
        st.markdown(_module_card(
            "Data Intelligence",
            "Implemented",
            "status-impl",
            "Dataset upload (CSV/TSV/XLSX/JSON) with source-content persistence, descriptive stats, missing-value analysis, numeric + categorical summaries, IQR outlier detection, correlation heatmaps, full dataset profiling, and Plotly visualizations. Stored datasets reusable by dataset_id.",
            "Automated EDA reports, anomaly detection models, dataset versioning.",
            "#ffb347",
        ), unsafe_allow_html=True)

    st.markdown("")
    c4, c5, c6 = st.columns(3)
    with c4:
        st.markdown(_module_card(
            "Translation",
            "Implemented",
            "status-impl",
            "Bidirectional JP ↔ EN translation with automatic LLM backend selection (Claude/Gemini) and dictionary fallback. 4 domain glossaries (business/tech/finance/legal), request-scoped glossaries + few-shot examples, batch translation, DOCX translation, and 4-factor translation quality evaluation.",
            "Neural fine-tuning, larger glossaries, layout-preserving PDF translation.",
            "#e04a1a",
        ), unsafe_allow_html=True)
    with c5:
        st.markdown(_module_card(
            "Document Intelligence",
            "Implemented",
            "status-impl",
            "Bilingual NLP reports in DOCX & PDF plus 5 official Japanese templates (履歴書, 職務経歴書, 請求書, 一般報告書, 送付状) with direct download. AI outline drafting, batch generation, DOCX translation, bilingual PDF merging, and company-analysis reports from uploaded/stored datasets.",
            "Template gallery upload, OCR for scanned documents, template version control.",
            "#c1121f",
        ), unsafe_allow_html=True)
    with c6:
        st.markdown(_module_card(
            "Machine Learning",
            "Implemented",
            "status-impl",
            "5-category Japanese text classification (LinearSVC + Sudachi TF-IDF), KMeans clustering, 6-intent detection (greeting/question/request/command/opinion/other), LDA topic modeling, confidence-gated auto-labeling, supervised BIO NER, and a CPU PyTorch Transformer classifier.",
            "Production-scale fine-tuning, multi-task learning, active learning loops.",
            "#ff4500",
        ), unsafe_allow_html=True)

    # ---- Live charts if backend online + history available ----
    if online and total_nlp > 0:
        st.markdown("")
        st.markdown("### Live Module Activity")
        ok_h, hist = fetch_history(limit=200)
        items = []
        if ok_h and isinstance(hist, dict):
            items = hist.get("history", []) or []
        if items:
            col_l, col_r = st.columns([3, 2])
            with col_l:
                rows = []
                for it in items:
                    ca = it.get("created_at")
                    if ca:
                        try:
                            ca_dt = datetime.fromisoformat(str(ca).replace("Z", ""))
                            rows.append({"ts": ca_dt, "tokens": int(it.get("token_count") or 0),
                                         "mode": it.get("mode", "C")})
                        except Exception:
                            pass
                if rows:
                    df = pd.DataFrame(rows).sort_values("ts")
                    df["rolling_avg"] = df["tokens"].rolling(min(7, len(df)), 1).mean()
                    fig = go.Figure()
                    fig.add_trace(go.Bar(x=df["ts"], y=df["tokens"], name="Tokens",
                                         marker_color="#cc4d00", opacity=0.8))
                    fig.add_trace(go.Scatter(x=df["ts"], y=df["rolling_avg"], mode="lines",
                                             name="Avg tokens", line=dict(color="#ff8c1a", width=2.5)))
                    fig.update_layout(title="Tokens Processed Over Time", template=_KANAME_TEMPLATE,
                                      height=340, showlegend=True, margin=dict(l=20, r=20, t=50, b=20))
                    st.plotly_chart(fig, use_container_width=True)
            with col_r:
                pos_all: Dict[str, int] = {}
                for it in items:
                    pc = it.get("pos_counts") or {}
                    if isinstance(pc, dict):
                        for k, v in pc.items():
                            pos_all[k] = pos_all.get(k, 0) + int(v or 0)
                if pos_all:
                    rows = sorted(pos_all.items(), key=lambda kv: -kv[1])[:10]
                    pdf = pd.DataFrame(rows, columns=["POS", "Count"])
                    pos_map = {"名詞": "Noun", "動詞": "Verb", "形容詞": "Adj",
                               "助詞": "Particle", "副詞": "Adv", "助動詞": "AuxV"}
                    pdf["POS_en"] = pdf["POS"].map(lambda x: pos_map.get(x, x))
                    fig2 = px.bar(
                        pdf, x="Count", y="POS_en", orientation="h", color="Count",
                        color_continuous_scale=_KANAME_SCALE,
                        title="Overall POS Distribution",
                    )
                    fig2.update_layout(height=340, template=_KANAME_TEMPLATE,
                                       margin=dict(l=20, r=20, t=50, b=20), showlegend=False)
                    st.plotly_chart(fig2, use_container_width=True)


# ---------------------------------------------------------------------------
# MODULE: JAPANESE NLP
# ---------------------------------------------------------------------------

STATUS_IMPLEMENTED = (
    '<span class="status-badge status-impl">Implemented — SudachiPy Active</span>'
)

SAMPLE_TEXTS = {
    "名詞・助詞（基本）": "私は日本語を勉強しています。",
    "長い複合名詞": "東京特許許可局局長は今日も良い天気です。",
    "固有名詞と動詞": "田中さんは京都大学で機械学習を研究しています。",
    "日常会話文": "昨日図書館で面白い本を読んだ後、カフェでコーヒーを飲みました。",
    "漢字・カタカナ混在": "AIと自然言語処理（NLP）の技術は急速に進歩しています。",
}

POS_LABEL_JP = {
    "名詞": "Noun", "動詞": "Verb", "形容詞": "Adjective", "副詞": "Adverb",
    "助詞": "Particle", "助動詞": "Auxiliary verb", "接尾辞": "Suffix",
    "接頭辞": "Prefix", "連体詞": "Adnominal", "感動詞": "Interjection",
    "記号": "Symbol", "補助記号": "Supplementary symbol", "空白": "Whitespace",
}

POS_COLOR = {
    "名詞": "#60a5fa", "動詞": "#fb7185", "形容詞": "#a78bfa", "副詞": "#22d3ee",
    "助詞": "#9aa3b2", "助動詞": "#eab308", "接尾辞": "#f59e0b",
    "記号": "#cbd5e1", "補助記号": "#cbd5e1",
}


def _render_backend_status_box():
    if not _REQUESTS_AVAILABLE:
        st.error("The 'requests' package is not installed.")
        return
    ok, msg = _request("GET", "/api/nlp/health")
    if ok and isinstance(msg, dict):
        st.success(
            f"NLP engine online: {msg.get('engine')}  ·  "
            f"Dictionary loaded: {'✅' if msg.get('dictionary_loaded') else '❌'}  ·  "
            f"Default mode: {msg.get('default_mode')}"
        )
    else:
        st.error(msg if isinstance(msg, str) else str(msg))


_JLPT_BADGE_COLOR = {
    "N5": "background:#166534;color:#fff;",
    "N4": "background:#15803d;color:#fff;",
    "N3": "background:#ca8a04;color:#111;",
    "N2": "background:#c2410c;color:#fff;",
    "N1": "background:#991b1b;color:#fff;",
}


def _jlpt_badge(jlpt: Optional[str]) -> str:
    if not jlpt:
        return ""
    style = _JLPT_BADGE_COLOR.get(jlpt, "background:#334155;color:#fff;")
    return (f'<span style="display:inline-block;font-size:0.7rem;padding:2px 6px;'
            f'border-radius:9999px;margin-left:4px;{style}">{jlpt}</span>')


def _render_token_chips(tokens: List[Dict]):
    html_parts = []
    for t in tokens:
        pos = t.get("pos_main", "")
        color = POS_COLOR.get(pos, "#94a3b8")
        reading = t.get("reading", "") or ""
        jlpt = t.get("jlpt_level")
        alt = t.get("alternatives") or {}
        romaji = alt.get("romaji")
        tooltip_parts = [pos, reading]
        if alt.get("dictionary_form"):
            tooltip_parts.append(f"辞書形: {alt['dictionary_form']}")
        if alt.get("synonym"):
            tooltip_parts.append(f"類義: {alt['synonym']}")
        if alt.get("keigo_honorific"):
            tooltip_parts.append(f"敬語: {alt['keigo_honorific']}")
        if romaji:
            tooltip_parts.append(f"romaji: {romaji}")
        if jlpt:
            tooltip_parts.append(f"JLPT: {jlpt}")
        title = " · ".join(x for x in tooltip_parts if x)
        chip_html = (
            f'<span title="{title}" '
            f'style="display:inline-block;margin:2px 4px 2px 0;padding:5px 10px;'
            f'border:1px solid {color}55;background:{color}22;color:#f2f2f5;'
            f'border-radius:8px;font-size:0.95rem;vertical-align:middle;">'
            f'<span style="font-weight:600;">{t.get("surface")}</span>'
        )
        if reading:
            chip_html += f' <small style="color:{color};font-weight:500;">{reading}</small>'
        if romaji:
            chip_html += f' <small style="color:#9a9aa8;font-size:0.7rem;">/{romaji}/</small>'
        chip_html += _jlpt_badge(jlpt)
        chip_html += "</span>"
        html_parts.append(chip_html)
    st.markdown("".join(html_parts), unsafe_allow_html=True)


def _tokens_dataframe(tokens: List[Dict]) -> pd.DataFrame:
    rows = []
    for t in tokens:
        alt = t.get("alternatives") or {}
        rows.append({
            "Surface": t.get("surface"),
            "JLPT": t.get("jlpt_level") or "",
            "POS (日本語)": t.get("pos_main"),
            "POS (EN)": POS_LABEL_JP.get(t.get("pos_main", ""), t.get("pos_main")),
            "Reading (読み)": t.get("reading"),
            "Romaji": alt.get("romaji", ""),
            "Dictionary Form": t.get("dictionary_form"),
            "Normalized": t.get("normalized_form"),
            "Synonym": alt.get("synonym", ""),
            "Keigo/敬語": alt.get("keigo_honorific", ""),
            "Full POS": ", ".join(t.get("pos") or []),
        })
    df = pd.DataFrame(rows)
    df.index = pd.RangeIndex(start=1, stop=len(df) + 1, name="#")
    return df


def _render_nlp_inputs():
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        sample_key = st.selectbox("Example sentences", options=list(SAMPLE_TEXTS.keys()), index=0)
    with col2:
        mode = st.selectbox("Split Mode", options=["C", "B", "A"], index=0,
                            help="C = longest (default) · B = middle · A = shortest")
    with col3:
        run_auto = st.checkbox("Auto-run on change", value=True)
    default_text = SAMPLE_TEXTS[sample_key]
    text = st.text_area("Enter Japanese text below", value=default_text, height=120,
                        placeholder="ここに日本語の文章を入力してください。")
    return text, mode, run_auto


def _render_analyze_summary(data: Dict):
    kpis = st.columns(5)
    kpis[0].metric("Tokens", data.get("token_count", 0))
    kpis[1].metric("Nouns (名詞)", len(data.get("noun_list", [])))
    kpis[2].metric("Verbs (動詞)", len(data.get("verb_list", [])))
    kpis[3].metric("Adjectives (形容詞)", len(data.get("adjective_list", [])))
    kpis[4].metric("POS Types", len(data.get("pos_counts", {})))

    aid = data.get("analysis_id")
    dur = data.get("duration_seconds")
    if aid or dur:
        meta_cols = st.columns([1, 1, 4])
        if aid:
            meta_cols[0].caption(f"Analysis ID: {aid}")
        if dur:
            meta_cols[1].caption(f"Duration: {dur}s")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Wakati-gaki (分かち書き)")
        st.code(data.get("wakati", ""), language=None)
        st.markdown("#### Kana-only reading (カナ読み)")
        st.code(data.get("kana_only_text", ""), language=None)
    with c2:
        st.markdown("#### POS Distribution")
        pos_df = pd.DataFrame(
            [{"POS (日本語)": k, "POS (English)": POS_LABEL_JP.get(k, k), "Count": v}
             for k, v in sorted(data.get("pos_counts", {}).items(), key=lambda x: -x[1])]
        )
        if not pos_df.empty:
            fig = px.bar(pos_df, x="POS (日本語)", y="Count", color="Count",
                         color_continuous_scale=_KANAME_SCALE,
                         title="POS Count by Category")
            fig.update_layout(height=280, template=_KANAME_TEMPLATE,
                              margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig, use_container_width=True)
            with st.expander("Show POS table"):
                st.dataframe(pos_df, use_container_width=True, hide_index=True)
        else:
            st.caption("No POS data.")


def render_nlp():
    st.markdown("## Japanese NLP Intelligence")
    st.markdown(STATUS_IMPLEMENTED, unsafe_allow_html=True)
    st.markdown("")

    _render_backend_status_box()

    with st.expander("About this module", expanded=False):
        st.markdown("""
            **Engine:** SudachiPy + SudachiDict-core (Works Applications)
            **Active features:**
            - Tokenization (分かち書き) with A/B/C split modes
            - Morphological analysis (形態素解析): POS, reading (読み), dictionary / normalized form
            - POS distribution counting + Plotly bar chart
            - Noun / Verb / Adjective extraction
            - Persisted history logging via `analysis_id`
            - **TF-IDF keyword extraction** (POS filtered, top-N ranked with frequency)
            - **Rule-based sentiment analysis** (JA lexicon ~200 words, matched chips + score −1 → +1)
            - **Named entities:** dictionary/POS extraction plus a supervised BIO model trainable from your span annotations
        """)

    st.markdown("### Input & Controls")
    text, mode, run_auto = _render_nlp_inputs()

    analyze_data = None
    if run_auto or st.button("Run NLP Analysis", type="primary", use_container_width=True):
        if not text.strip():
            st.warning("Enter at least one character of Japanese text.")
        else:
            with st.spinner("Running SudachiPy morphological analysis via KANAME backend..."):
                ok, msg = _request("POST", "/api/nlp/analyze", {"text": text, "mode": mode})
                if ok:
                    analyze_data = msg
                else:
                    st.error(f"Analysis failed:\n{msg}")

    if analyze_data is not None:
        st.markdown("---")
        _render_analyze_summary(analyze_data)

        st.markdown("---")
        st.markdown("### Tokens (Surface + Reading chips)")
        _render_token_chips(analyze_data.get("tokens", []))

        st.markdown("")
        with st.expander("See extracted word lists"):
            col_a, col_b, col_c = st.columns(3)
            col_a.markdown("**Nouns (名詞)**")
            col_a.dataframe(pd.DataFrame({"Noun": analyze_data.get("noun_list", [])}),
                            use_container_width=True, hide_index=True)
            col_b.markdown("**Verbs (動詞)** — dictionary forms")
            col_b.dataframe(pd.DataFrame({"Verb": analyze_data.get("verb_list", [])}),
                            use_container_width=True, hide_index=True)
            col_c.markdown("**Adjectives (形容詞)** — dictionary forms")
            col_c.dataframe(pd.DataFrame({"Adjective": analyze_data.get("adjective_list", [])}),
                            use_container_width=True, hide_index=True)

        st.markdown("")
        st.markdown("### Full Morphological Table")
        st.dataframe(_tokens_dataframe(analyze_data.get("tokens", [])), use_container_width=True)

        with st.expander("Raw JSON response from FastAPI backend"):
            st.json(analyze_data)

    # ---- Keywords + Sentiment (independent reusable section) ----
    st.markdown("")
    st.markdown("---")
    st.markdown("### Keywords · Sentiment · Named Entities")
    st.caption("Runs keyword, rule-based sentiment, and dictionary/POS-based entity analysis on the same input text.")
    col_kws_topn, col_kws_mode, col_kws_btn = st.columns([1, 1, 2])
    with col_kws_topn:
        top_n = st.slider("Top-N keywords", min_value=5, max_value=30, value=15, step=1, key="nlp_kw_topn")
    with col_kws_mode:
        kw_mode = st.selectbox("Sudachi mode (keywords)", ["C", "B", "A"], index=0, key="nlp_kw_mode")
    with col_kws_btn:
        st.markdown("<div style='height:26px'></div>", unsafe_allow_html=True)
        run_kw = st.button("Run NLP Enrichment", type="secondary", use_container_width=True)

    if (run_kw or analyze_data is not None) and text.strip() and _REQUESTS_AVAILABLE:
        c_kw, c_sen = st.columns(2)
        with c_kw:
            with st.spinner("Extracting keywords via POST /api/nlp/keywords ..."):
                ok_kw, r_kw = _request("POST", "/api/nlp/keywords",
                                       payload={"text": text, "mode": kw_mode, "top_n": top_n},
                                       timeout=60)
            if ok_kw and isinstance(r_kw, dict):
                st.markdown(f"**Keywords** — {r_kw.get('keyword_count', 0)} extracted")
                items = r_kw.get("keywords") or []
                if items and _PLOTLY_AVAILABLE:
                    df_kw = pd.DataFrame(items)
                    df_kw_display = df_kw.copy()
                    df_kw_display["Romaji"] = [
                        (it.get("alternatives") or {}).get("romaji", "")
                        for it in items
                    ]
                    df_kw_display["JLPT"] = [it.get("jlpt_level") or "" for it in items]
                    df_kw_display["類義語/Synonym"] = [
                        (it.get("alternatives") or {}).get("synonym", "")
                        for it in items
                    ]
                    df_kw_plot = df_kw.sort_values("score", ascending=True).tail(20)
                    fig = px.bar(df_kw_plot, x="score", y="keyword", orientation="h",
                                 title="Keyword score (TF-IDF style)",
                                 color="score", color_continuous_scale=_KANAME_SCALE,
                                 custom_data=[
                                     [it.get("jlpt_level") or "—" for it in df_kw_plot.to_dict("records")],
                                     [(it.get("alternatives") or {}).get("romaji","") for it in df_kw_plot.to_dict("records")],
                                 ],
                                 hover_data=["frequency", "pos_hint"])
                    fig.update_traces(hovertemplate="<b>%{y}</b><br>Score: %{x:.3f}<br>JLPT: %{customdata[0]}<br>Romaji: %{customdata[1]}")
                    fig.update_layout(height=300+8*len(df_kw_plot), margin=dict(l=0, r=10, t=40, b=0))
                    st.plotly_chart(fig, use_container_width=True)
                    with st.expander(f"📑 Keyword details ({len(items)} rows) w/ JLPT + Romaji + Synonyms"):
                        st.dataframe(
                            df_kw_display[["keyword", "JLPT", "score", "frequency", "pos_hint",
                                           "Romaji", "類義語/Synonym"]],
                            use_container_width=True, hide_index=True,
                        )
                elif items:
                    df_kw_display = pd.DataFrame([{
                        "keyword": it.get("keyword"),
                        "JLPT": it.get("jlpt_level") or "",
                        "score": it.get("score"),
                        "frequency": it.get("frequency"),
                        "Romaji": (it.get("alternatives") or {}).get("romaji", ""),
                        "Synonym": (it.get("alternatives") or {}).get("synonym", ""),
                    } for it in items])
                    st.dataframe(df_kw_display, use_container_width=True, hide_index=True)
                else:
                    st.caption("No keywords found for the given input.")
                with st.expander("Raw keywords JSON"):
                    st.json(r_kw)
            elif not ok_kw:
                st.error(f"Keywords failed: {r_kw}")

        with c_sen:
            with st.spinner("Analyzing sentiment via POST /api/nlp/sentiment ..."):
                ok_s, r_s = _request("POST", "/api/nlp/sentiment",
                                     payload={"text": text, "mode": kw_mode}, timeout=60)
            if ok_s and isinstance(r_s, dict):
                score = float(r_s.get("score", 0.0))
                label = r_s.get("label", "neutral")
                confidence = float(r_s.get("confidence", 0.0))
                sm1, sm2, sm3 = st.columns(3)
                sm1.metric("Score", f"{score:+.2f}")
                sm2.metric("Label", label)
                sm3.metric("Confidence", f"{confidence*100:.0f}%")

                if _PLOTLY_AVAILABLE:
                    df_gauge = pd.DataFrame({
                        "direction": ["Negative ←", "Neutral", "→ Positive"],
                        "weight": [max(0, -score), (1 - abs(score)), max(0, score)],
                    })
                    fig_s = px.bar(df_gauge, x="weight", y=["gauge"]*3, color="direction", orientation="h",
                                   color_discrete_sequence=["#ff4d4d", "#9a9aa8", "#4ade80"],
                                   title="Sentiment distribution −1.0 ↔ +1.0",
                                   range_x=[0, 1.15],
                                   labels={"weight": "", "y": ""})
                    fig_s.update_layout(barmode="stack", height=160, showlegend=True,
                                        margin=dict(l=0, r=10, t=40, b=0),
                                        yaxis=dict(visible=False, showticklabels=False))
                    st.plotly_chart(fig_s, use_container_width=True)

                pos_matches = r_s.get("positive_matches") or []
                neg_matches = r_s.get("negative_matches") or []
                pos_jlpt = r_s.get("positive_jlpt") or {}
                neg_jlpt = r_s.get("negative_jlpt") or {}
                alt_map = r_s.get("matches_alternatives") or {}
                st.caption(f"Matched lexicon: ✅ +{len(pos_matches)} positives  ·  ❌ −{len(neg_matches)} negatives")
                if pos_matches:
                    with st.expander(f"✅ Positive words ({len(pos_matches)})"):
                        chip_parts = []
                        seen = set()
                        for w in pos_matches:
                            if w in seen:
                                continue
                            seen.add(w)
                            jlpt = pos_jlpt.get(w)
                            alt = alt_map.get(w) or {}
                            title_parts = []
                            if alt.get("reading"):
                                title_parts.append(f"読み: {alt['reading']}")
                            if alt.get("romaji"):
                                title_parts.append(f"romaji: {alt['romaji']}")
                            if alt.get("synonym"):
                                title_parts.append(f"類義: {alt['synonym']}")
                            title = ("JLPT: " + jlpt + " · " if jlpt else "") + " · ".join(title_parts)
                            chip_html = (
                                f"<span title='{title}' class='status-badge status-impl' "
                                f"style='font-size:0.8rem;margin:3px 6px 3px 0;'>{w}"
                            )
                            if alt.get("romaji"):
                                chip_html += f" <small style='opacity:.7'>/{alt['romaji']}/</small>"
                            chip_html += _jlpt_badge(jlpt)
                            chip_html += "</span>"
                            chip_parts.append(chip_html)
                        st.markdown("".join(chip_parts), unsafe_allow_html=True)
                if neg_matches:
                    with st.expander(f"❌ Negative words ({len(neg_matches)})"):
                        chip_parts = []
                        seen = set()
                        for w in neg_matches:
                            if w in seen:
                                continue
                            seen.add(w)
                            jlpt = neg_jlpt.get(w)
                            alt = alt_map.get(w) or {}
                            title_parts = []
                            if alt.get("reading"):
                                title_parts.append(f"読み: {alt['reading']}")
                            if alt.get("romaji"):
                                title_parts.append(f"romaji: {alt['romaji']}")
                            if alt.get("synonym"):
                                title_parts.append(f"類義: {alt['synonym']}")
                            title = ("JLPT: " + jlpt + " · " if jlpt else "") + " · ".join(title_parts)
                            chip_html = (
                                f"<span title='{title}' style='background:#991b1b;color:#fff;font-size:0.8rem;"
                                f"margin:3px 6px 3px 0;padding:4px 10px;border-radius:9999px;display:inline-block;'>{w}"
                            )
                            if alt.get("romaji"):
                                chip_html += f" <small style='opacity:.7'>/{alt['romaji']}/</small>"
                            chip_html += _jlpt_badge(jlpt)
                            chip_html += "</span>"
                            chip_parts.append(chip_html)
                        st.markdown("".join(chip_parts), unsafe_allow_html=True)
                with st.expander("Raw sentiment JSON"):
                    st.json(r_s)
            elif not ok_s:
                st.error(f"Sentiment failed: {r_s}")

        with st.expander("Named Entity Recognition", expanded=True):
            with st.spinner("Extracting named entities via POST /api/nlp/entities ..."):
                ok_ner, r_ner = _request(
                    "POST",
                    "/api/nlp/entities",
                    payload={"text": text, "mode": kw_mode},
                    timeout=60,
                )
            if ok_ner and isinstance(r_ner, dict):
                st.caption(r_ner.get("method", ""))
                entities = r_ner.get("entities") or []
                if entities:
                    st.dataframe(
                        pd.DataFrame(entities)[["text", "label", "start", "end", "reading"]],
                        use_container_width=True,
                        hide_index=True,
                    )
                else:
                    st.caption("No dictionary-tagged named entities found.")
                with st.expander("Raw entity JSON"):
                    st.json(r_ner)
            else:
                if isinstance(r_ner, str) and "HTTP 404" in r_ner:
                    st.warning(
                        "The backend at "
                        f"{BACKEND_URL} does not have this NER route yet. "
                        "It is running an older version. Stop the existing backend, "
                        "then restart it from the project root with: "
                        "`python -m uvicorn backend.main:app --reload --port 8000`."
                    )
                else:
                    st.error(f"Named entity extraction failed: {r_ner}")

            ok_model, model_info = _request("GET", "/api/nlp/entities/model")
            if ok_model and isinstance(model_info, dict) and model_info.get("trained"):
                st.caption(
                    "Supervised model ready: "
                    f"{', '.join(model_info.get('entity_labels') or [])}. "
                    "Training accuracy is not a held-out accuracy measure."
                )
            with st.expander("Train supervised NER from annotations", expanded=False):
                st.caption(
                    "Provide at least two labeled examples. start/end are exact Python string offsets "
                    "and must align to Sudachi token boundaries."
                )
                annotation_json = st.text_area(
                    "Annotated training examples (JSON)",
                    value=DEFAULT_NER_ANNOTATIONS,
                    height=150,
                    key="ner_training_annotations_v2",
                )
                if st.button("Train model and predict current text", key="ner_train_button"):
                    try:
                        training_examples = json.loads(
                            annotation_json.strip() or DEFAULT_NER_ANNOTATIONS
                        )
                        if not isinstance(training_examples, list):
                            raise ValueError("Expected a JSON array of annotated examples.")
                        ok_train, training_result = _request(
                            "POST",
                            "/api/nlp/entities/train",
                            payload={"examples": training_examples, "mode": kw_mode, "save": True},
                            timeout=180,
                        )
                        if not ok_train:
                            st.error(str(training_result))
                        else:
                            st.success(
                                f"Trained on {training_result.get('training_examples')} examples. "
                                "Use a separate labeled set to measure generalization."
                            )
                            ok_predict, prediction = _request(
                                "POST",
                                "/api/nlp/entities/predict",
                                payload={"text": text, "mode": kw_mode},
                                timeout=60,
                            )
                            if ok_predict and isinstance(prediction, dict):
                                predicted_entities = prediction.get("entities") or []
                                if predicted_entities:
                                    st.dataframe(
                                        pd.DataFrame(predicted_entities)[
                                            ["text", "label", "start", "end", "confidence"]
                                        ],
                                        use_container_width=True,
                                        hide_index=True,
                                    )
                                else:
                                    st.caption("The supervised model found no entities in the current text.")
                            else:
                                st.error(str(prediction))
                    except (json.JSONDecodeError, ValueError) as exc:
                        st.error(f"Invalid annotation JSON: {exc}")

    # ---- History ----
    st.markdown("")
    st.markdown("### Recent Analysis History")
    online, _, _ = backend_online()
    if online:
        ok_h, hist = fetch_history(limit=50)
        if ok_h and isinstance(hist, dict):
            items = hist.get("history", []) or []
            if items:
                table_rows = []
                for it in items:
                    table_rows.append({
                        "Created": it.get("created_at"),
                        "Mode": it.get("mode"),
                        "Tokens": it.get("token_count"),
                        "Nouns": len(it.get("noun_list") or []),
                        "Preview": it.get("input_text"),
                    })
                hdf = pd.DataFrame(table_rows)
                st.dataframe(hdf, use_container_width=True, hide_index=True)
            else:
                st.caption("No history yet. Run an analysis above!")
        else:
            st.caption(f"(No history available — DB tables not created?) Message: {hist}")
    else:
        st.caption("Start the FastAPI backend to view history.")


# ---------------------------------------------------------------------------
# MODULE: VOICE INTELLIGENCE
# ---------------------------------------------------------------------------

def render_voice():
    st.markdown("## Voice Intelligence")
    st.markdown('<span class="status-badge status-impl">Implemented — faster-whisper (CTranslate2)</span>',
                unsafe_allow_html=True)
    st.markdown("")

    if _REQUESTS_AVAILABLE:
        ok_v, msg = _request("GET", "/api/voice/health")
        if ok_v and isinstance(msg, dict):
            st.success(
                f"Voice engine: {msg.get('engine')}  ·  "
                f"Available: {'✅' if msg.get('engine_available') else '❌'}  ·  "
                f"Default model: {msg.get('default_model')}  ·  "
                f"Default language: {msg.get('default_language')}"
            )
            if not msg.get("engine_available") and msg.get("detail"):
                st.warning(f"Detail: {msg.get('detail')}")
        else:
            st.error(msg if isinstance(msg, str) else "Backend offline")

    with st.expander("Module Description & Capabilities", expanded=False):
        st.markdown("""
            **Engine:** faster-whisper (CTranslate2 optimized Whisper) — tiny → base → small → medium → large-v3
            **Active features:**
            - Audio upload + Microphone recording (via dedicated Gradio interface)
            - Multiple model sizes with configurable beam search, temperature, VAD filter
            - Japanese language auto-detection or forced selection
            - Per-segment timestamps, language probabilities, warnings
            - VoiceJob persistence (SQLite)
            - Bonus `transcribe-then-nlp` endpoint: STT → SudachiPy morphological pipeline
            - Voice commands are parsed into safe, non-executing action suggestions
            - Optional host-engine TTS and pretrained pyannote diarization adapters

            **Dedicated Gradio interface:** http://localhost:7860 (run separately)
        """)

    col1, col2 = st.columns([2, 3])
    with col1:
        st.markdown("### 🎙 Upload & Transcribe (Direct)")
        audio_file = st.file_uploader(
            "Upload audio (wav / mp3 / m4a / flac / webm)",
            type=["wav", "mp3", "m4a", "flac", "webm", "ogg"],
        )
        with st.form("voice_form"):
            m_col1, m_col2 = st.columns(2)
            with m_col1:
                model_size = st.selectbox("Model", ["tiny", "base", "small", "medium", "large-v3"], index=1)
                language = st.selectbox("Language", ["auto", "ja", "en"], index=0)
                sudachi_mode = st.selectbox("Post NLP Mode", ["C", "B", "A"], index=0, key="vsm")
            with m_col2:
                beam_size = st.slider("Beam Size", 1, 10, 5)
                temperature = st.slider("Temperature", 0.0, 1.0, 0.0, step=0.05)
                vad_filter = st.checkbox("VAD Filter", value=True)
            run_both = st.form_submit_button("🔊 Transcribe (and run NLP)", type="primary",
                                             use_container_width=True)

        if audio_file is not None and run_both:
            fname = audio_file.name
            fbytes = audio_file.getvalue()
            if not _REQUESTS_AVAILABLE:
                st.error("requests not installed")
            else:
                with st.spinner(f"Transcribing with {model_size} model via KANAME backend..."):
                    ok, resp = _request(
                        "POST", "/api/voice/transcribe-then-nlp",
                        files={"file": (fname, fbytes)},
                        data={
                            "model_size": model_size, "language": language,
                            "beam_size": str(beam_size), "temperature": str(temperature),
                            "device": "auto", "vad_filter": "true" if vad_filter else "false",
                            "sudachi_mode": sudachi_mode,
                        },
                        timeout=600,
                    )
                if ok and isinstance(resp, dict):
                    stt = resp.get("stt") or {}
                    nlp = resp.get("nlp") or None
                    st.markdown("#### ✅ Transcription")
                    st_cols = st.columns(4)
                    st_cols[0].metric("Language", stt.get("language", "?"))
                    st_cols[1].metric("Duration (s)", round(float(stt.get("duration", 0) or 0), 2))
                    st_cols[2].metric("Segments", len(stt.get("segments") or []))
                    pct = float(stt.get("language_probability") or 0)
                    st_cols[3].metric("Lang prob", f"{pct:.0%}")
                    st.text_area("Transcript", value=stt.get("text") or "", height=200)
                    if stt.get("text") and _REQUESTS_AVAILABLE:
                        with st.expander("Voice command interpretation", expanded=False):
                            ok_cmd, cmd = _request(
                                "POST", "/api/voice/commands/parse",
                                payload={"text": stt.get("text", "")},
                            )
                            if ok_cmd and isinstance(cmd, dict):
                                st.write(f"**Intent:** {cmd.get('intent')} · **Suggested action:** {cmd.get('action')}")
                                st.caption(cmd.get("note", ""))
                            else:
                                st.warning(str(cmd))
                    if stt.get("warnings"):
                        for w in stt["warnings"]:
                            st.warning("⚠ " + w)

                    if nlp:
                        st.markdown("#### 🧠 Post-Transcription NLP (SudachiPy)")
                        nk = st.columns(4)
                        nk[0].metric("Tokens", nlp.get("token_count", 0))
                        nk[1].metric("Nouns", len(nlp.get("noun_list") or []))
                        nk[2].metric("Verbs", len(nlp.get("verb_list") or []))
                        nk[3].metric("Adjectives", len(nlp.get("adjective_list") or []))
                        st.code(nlp.get("wakati", ""), language=None)
                        with st.expander("Raw NLP result"):
                            st.json(nlp)
                else:
                    st.error(f"Transcription failed: {resp}")

    if audio_file is not None and st.button("Run pretrained speaker diarization", key="voice_diarize"):
        if not _REQUESTS_AVAILABLE:
            st.error("requests not installed")
        else:
            with st.spinner("Transcribing and running the optional pyannote diarization model..."):
                ok_diar, diar = _request(
                    "POST", "/api/voice/diarize",
                    files={"file": (audio_file.name, audio_file.getvalue(), audio_file.type)},
                    data={"model_size": model_size, "language": language, "vad_filter": str(vad_filter).lower()},
                    timeout=900,
                )
            if ok_diar and isinstance(diar, dict):
                st.dataframe(pd.DataFrame(diar.get("aligned_segments") or []),
                             use_container_width=True, hide_index=True)
            else:
                st.warning(str(diar))

    with col2:
        st.markdown("### 🎛 Gradio Interface (Microphone + Full Controls)")
        st.info(
            "For **live microphone** recording, advanced parameter tuning, and segmented output, "
            "use the dedicated Gradio app on port 7860.\n\n"
            "Run it in a separate terminal:\n"
            "```\npython frontend/gradio_app/app.py\n```"
        )
        st.markdown("""
            **Gradio features:**
            - 🎤 Microphone input (real-time)
            - 📁 Audio upload (any format supported by PyAV)
            - ⚙ Tune model size, language, beam, temp, VAD, initial prompt
            - 🧠 After STT: Run SudachiPy NLP directly
            - 📋 Copy-to-clipboard output
        """)
        st.markdown("### Text-to-Speech")
        tts_text = st.text_area("Text to speak", height=90, key="voice_tts_text")
        tts_language = st.selectbox("TTS voice language", ["ja", "en"], key="voice_tts_language")
        if st.button("Synthesize Speech", key="voice_tts_run") and tts_text.strip() and _REQUESTS_AVAILABLE:
            try:
                tts_response = requests.post(
                    f"{BACKEND_URL}/api/voice/tts",
                    json={"text": tts_text, "language": tts_language},
                    timeout=120,
                )
                if tts_response.ok:
                    st.audio(tts_response.content, format="audio/wav")
                else:
                    st.warning(f"HTTP {tts_response.status_code}: {tts_response.text[:300]}")
            except Exception as exc:
                st.error(f"TTS request failed: {exc}")

    st.markdown("")
    st.markdown("### Recent Voice Jobs")
    online, _, _ = backend_online()
    if online and _REQUESTS_AVAILABLE:
        ok_v, resp = _request("GET", "/api/voice/jobs?limit=30")
        if ok_v and isinstance(resp, dict):
            jobs = resp.get("jobs") or []
            if jobs:
                rows = []
                for j in jobs:
                    rows.append({
                        "Created": j.get("created_at"),
                        "File": j.get("source_filename"),
                        "Model": j.get("model_size"),
                        "Lang": j.get("detected_language"),
                        "Lang pct": f"{float(j.get('language_probability') or 0):.0%}",
                        "Duration (s)": round(float(j.get("duration_seconds") or 0), 2),
                        "Segments": j.get("segment_count"),
                        "Preview": (j.get("transcription_text") or "")[:80],
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            else:
                st.caption("No voice jobs yet. Try running a transcription above!")
        else:
            st.caption("(No jobs available — DB tables may not be created. Run init_db.py first.)")
    else:
        st.caption("Start the FastAPI backend to view history.")


# ---------------------------------------------------------------------------
# MODULE: DATA INTELLIGENCE
# ---------------------------------------------------------------------------

def _human_size(n: int) -> str:
    if not n:
        return "0 B"
    for unit in ["B", "KB", "MB", "GB"]:
        if abs(n) < 1024.0:
            return f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} TB"


def render_data():
    st.markdown("## Data Intelligence — Interactive Dashboard")
    st.markdown('<span class="status-badge status-impl">Implemented — pandas + NumPy + Plotly + Interactive Streamlit Widgets</span>',
                unsafe_allow_html=True)
    st.markdown("")

    if _REQUESTS_AVAILABLE:
        ok, msg = _request("GET", "/api/data/health")
        if ok and isinstance(msg, dict):
            st.success(
                f"Engine: {msg.get('engine')}  ·  "
                f"Supported: {', '.join(msg.get('supported_formats') or [])}"
            )
        else:
            st.error(msg if isinstance(msg, str) else "Backend offline")

    with st.expander("Module Description", expanded=False):
        st.markdown("""
            **Interactive dashboard (8 widgets):**
            1. 📤 Upload / sample dataset picker
            2. 🔍 Column inclusion filters (numeric, categorical)
            3. 🎨 Chart type switcher (Histogram / Scatter / Bar / Box / Violin / Heatmap)
            4. 📏 IQR outlier sensitivity slider (k multiplier, 0.1 mild → 10 extreme)
            5. 🌈 Correlation heatmap colormap selector
            6. 🔎 Editable data table (`st.data_editor` `on_select="rerun"`)
            7. 📋 Selection summary & per-selected-row describe()
            8. 💾 Download CSV (filtered) + PNG (last figure)
        """)

    # -------- WIDGET 1: Upload / sample picker --------
    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown("### 1️⃣ Dataset Source")
        file = st.file_uploader(
            "CSV / TSV / XLSX / XLS / JSON",
            type=["csv", "tsv", "xlsx", "xls", "json"],
            key="dash_file",
        )
        sample_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "datasets", "japanese_text_corpus_sample.csv",
        )
        use_sample = st.checkbox(
            "Use bundled sample dataset (japanese_text_corpus_sample.csv)",
            value=file is None,
            key="dash_use_sample",
        )
        file_bytes = None
        filename = None
        if file is None and use_sample and os.path.exists(sample_path):
            with open(sample_path, "rb") as f:
                file_bytes = f.read()
            filename = os.path.basename(sample_path)
        elif file is not None:
            file_bytes = file.getvalue()
            filename = file.name

        # Optional: history dataset_id picker (W1 additional option)
        history_pick = None
        if _REQUESTS_AVAILABLE:
            try:
                ok_h, resp_h = _request("GET", "/api/data/datasets?limit=20")
                if ok_h and isinstance(resp_h, dict) and resp_h.get("datasets"):
                    opts = [(d.get("dataset_id"), f"{d.get('dataset_id')[:8]}… {d.get('filename') or ''} ({d.get('rows')}×{d.get('columns')})")
                            for d in resp_h["datasets"] if d.get("dataset_id")]
                    if opts:
                        opts.insert(0, (None, "— Use uploaded / sample —"))
                        history_pick = st.selectbox(
                            "Or pick a stored dataset",
                            [o[0] for o in opts],
                            index=0,
                            format_func=lambda v: dict(opts).get(v, str(v)),
                            key="dash_hist_pick",
                        )
            except Exception:
                pass

    # -------- WIDGET 4 (needs to be early because it controls analysis) + WIDGET 5 --------
    with c2:
        st.markdown("### 2️⃣ Dashboard Controls")
        chart_type = st.selectbox(
            "Chart type",
            ["Histogram", "Scatter", "Bar", "Box", "Violin", "Correlation Heatmap"],
            index=0,
            key="dash_chart_type",
        )
        iqr_k = st.slider(
            "IQR outlier sensitivity (k)",
            0.1, 10.0, 1.5, 0.1,
            help="Lower = more outliers flagged as extreme, higher = fewer (3 = extreme anomalies only)",
            key="dash_iqr_k",
        )
        colormap = st.selectbox(
            "Heatmap colormap",
            ["Blues", "Reds", "Greens", "Viridis", "Plasma", "RdBu", "Portland", "Cividis", "Magma", "YlOrRd"],
            index=9,
            key="dash_cmap",
        )
        sample_rows = st.slider("Sample records max shown in table", 1, 100, 20, key="dash_rows")

    # Load dataset once into a pandas DataFrame we can use throughout
    local_df: Optional[pd.DataFrame] = None
    if file_bytes is not None:
        try:
            encodings = ["utf-8", "utf-8-sig", "shift_jis", "cp932"]
            last_err = None
            if filename and filename.lower().endswith(".csv"):
                for enc in encodings:
                    try:
                        local_df = pd.read_csv(io.BytesIO(file_bytes), encoding=enc)
                        break
                    except Exception as e:
                        last_err = e
                if local_df is None and last_err:
                    raise last_err
            elif filename and filename.lower().endswith(".tsv"):
                for enc in encodings:
                    try:
                        local_df = pd.read_csv(io.BytesIO(file_bytes), encoding=enc, sep="\t")
                        break
                    except Exception as e:
                        last_err = e
                if local_df is None and last_err:
                    raise last_err
            elif filename and filename.lower().endswith(".json"):
                local_df = pd.read_json(io.BytesIO(file_bytes))
            elif filename and (filename.lower().endswith(".xlsx") or filename.lower().endswith(".xls")):
                local_df = pd.read_excel(io.BytesIO(file_bytes))
        except Exception as e:
            st.error(f"Failed to parse dataset: {e}")

    # Try pulling from stored dataset_id if selected & upload empty
    if local_df is None and history_pick and _REQUESTS_AVAILABLE:
        try:
            ok_d, resp_d = _request("GET", f"/api/data/datasets/{history_pick}")
            if ok_d and isinstance(resp_d, dict) and "dataset" in resp_d and resp_d["dataset"] is not None:
                rows = resp_d["dataset"].get("sample_records")
                if rows:
                    local_df = pd.DataFrame(rows)
        except Exception:
            pass

    if local_df is None:
        st.info("Upload a dataset above, pick a stored dataset, or check the bundled sample CSV checkbox.")
        return

    with st.expander("ML Dataset Quality Profile", expanded=False):
        st.caption("Flags duplicate rows, missingness, constant/high-cardinality columns, skew, and IQR outliers.")
        if st.button("Profile Dataset", key="dataset_profile_run") and _REQUESTS_AVAILABLE:
            if file_bytes is None and not history_pick:
                st.warning("Choose a file or a stored dataset before profiling.")
            else:
                ok_profile, profile_response = _request(
                    "POST",
                    "/api/data/profile",
                    files=(
                        {"file": (filename or "dataset.csv", file_bytes, "application/octet-stream")}
                        if file_bytes is not None else None
                    ),
                    data={
                        **({"dataset_id": history_pick} if file_bytes is None else {}),
                        "iqr_k": str(iqr_k),
                    },
                    timeout=180,
                )
                profile = profile_response.get("profile") if ok_profile and isinstance(profile_response, dict) else None
                if profile:
                    p1, p2, p3 = st.columns(3)
                    p1.metric("Rows", profile.get("rows", 0))
                    p2.metric("Duplicate rows", profile.get("duplicate_rows", 0))
                    p3.metric("Profiled columns", profile.get("columns", 0))
                    st.dataframe(pd.DataFrame(profile.get("profiled_columns") or []),
                                 use_container_width=True, hide_index=True)
                    for recommendation in profile.get("recommendations", []):
                        st.info(recommendation)
                else:
                    st.error(str(profile_response))

    st.markdown("---")

    # Auto detect numeric / categorical / datetime columns
    numeric_cols_available = list(local_df.select_dtypes(include="number").columns)
    cat_cols_available = list(local_df.select_dtypes(exclude="number").columns)

    # -------- WIDGET 2: Column inclusion filters --------
    with st.expander("🛠  2️⃣ Column Inclusion Filters", expanded=False):
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            use_numeric = st.multiselect(
                "Numeric columns to include",
                options=numeric_cols_available,
                default=numeric_cols_available,
                key="dash_use_num_cols",
            )
        with col_f2:
            use_cat = st.multiselect(
                "Categorical columns to include",
                options=cat_cols_available,
                default=cat_cols_available[: min(5, len(cat_cols_available))],
                key="dash_use_cat",
            )

    keep_cols = list(dict.fromkeys(use_numeric + use_cat))  # preserve order, de-dup
    if not keep_cols:
        st.warning("No columns selected. Select at least one column above.")
        return
    view_df = local_df[keep_cols].copy()

    # -------- Summary metrics --------
    summary_c = st.columns(5)
    summary_c[0].metric("Rows", f"{len(view_df):,}")
    summary_c[1].metric("Columns", len(keep_cols))
    summary_c[2].metric("Numeric cols", len(use_numeric))
    summary_c[3].metric("Categorical cols", len(use_cat))
    total_missing = int(view_df.isna().sum().sum())
    summary_c[4].metric("Missing Cells", f"{total_missing:,}")

    col_n, col_c = st.columns(2)
    with col_n:
        st.markdown("**Numeric:** " + (", ".join(use_numeric) if use_numeric else "—"))
    with col_c:
        st.markdown("**Categorical:** " + (", ".join(use_cat) if use_cat else "—"))

    with st.expander("Column dtypes & missing values"):
        mc = pd.DataFrame([
            {"Column": c,
             "Dtype": str(local_df[c].dtype),
             "Missing": int(local_df[c].isna().sum()),
             "Selected": "✅" if c in keep_cols else "—"}
            for c in local_df.columns
        ])
        st.dataframe(mc, use_container_width=True, hide_index=True)

    # -------- WIDGET 6: st.data_editor with on_select rerun --------
    st.markdown("### 3️⃣ Editable Data Table (select rows → live summary below)")
    display_df = view_df.head(sample_rows).reset_index(drop=True)
    edited = st.data_editor(
        display_df,
        use_container_width=True,
        hide_index=False,
        num_rows="dynamic",
        key="dash_editor",
        on_select="rerun",
        height=360,
    )

    # -------- WIDGET 7: Selection summary + describe() --------
    st.markdown("### 4️⃣ Selected Row Summary")
    sel = st.session_state.get("dash_editor", {}).get("selection") if hasattr(st.session_state, "dash_editor") else None
    try:
        sel_idx = list(edited.selection.rows)
    except Exception:
        sel_idx = []
    sel_count = len(sel_idx)
    sc1, sc2, sc3, sc4 = st.columns(4)
    sc1.metric("Selected rows", sel_count)
    sc2.metric("Total rows displayed", len(display_df))
    sc3.metric("Total numeric values",
               int(display_df[use_numeric].count().sum()) if use_numeric else 0)
    sc4.metric("Selection unique combos (cat)",
               len(display_df.loc[sel_idx, use_cat].drop_duplicates())
               if sel_idx and use_cat else 0)
    if sel_idx:
        sel_df = display_df.loc[sel_idx]
        with st.expander(f"Selected rows ({len(sel_df)})", expanded=True):
            st.dataframe(sel_df, use_container_width=True)
        if use_numeric:
            try:
                st.markdown("**Numeric describe() on selected rows:**")
                desc = sel_df[use_numeric].describe().round(4)
                st.dataframe(desc, use_container_width=True)
            except Exception as e:
                st.caption(f"describe unavailable: {e}")
    else:
        st.caption("Click rows in the table above to see live summaries here.")

    # -------- Main Plot (driven by WIDGET 3 chart-type switcher) --------
    st.markdown("### 5️⃣ Plotly Visualizations — Interactive")
    last_figure = None
    p1, p2 = st.columns([3, 1])
    if chart_type == "Histogram":
        if use_numeric:
            with p2:
                h_col = st.selectbox("X column", use_numeric, index=0, key="dash_hist_x")
                h_nbins = st.slider("Bins", 5, 100, 20, key="dash_hist_bins")
                color_col = st.selectbox("Color by", ["(none)"] + use_cat, index=0, key="dash_hist_color")
                args = {"x": h_col, "nbins": h_nbins, "title": f"Distribution — {h_col}"}
                if color_col != "(none)":
                    args["color"] = color_col
            fig = px.histogram(view_df, **args)
            fig.update_layout(template=_KANAME_TEMPLATE, height=420, margin=dict(l=20, r=20, t=60, b=20))
            st.plotly_chart(fig, use_container_width=True, key="dash_hist_fig")
            last_figure = fig
        else:
            st.info("Select at least one numeric column to plot a histogram.")

    elif chart_type == "Scatter":
        if len(use_numeric) >= 2:
            with p2:
                sx = st.selectbox("X", use_numeric, index=0, key="dash_sc_x")
                sy = st.selectbox("Y", use_numeric, index=min(1, len(use_numeric)-1), key="dash_sc_y")
                szo = st.selectbox("Size", ["(none)"] + use_numeric, index=0, key="dash_sc_size")
                cb = st.selectbox("Color", ["(none)"] + use_cat, index=0, key="dash_sc_color")
                args = {"x": sx, "y": sy}
                if cb != "(none)":
                    args["color"] = cb
                if szo != "(none)":
                    args["size"] = szo
            fig = px.scatter(view_df, **args, title=f"{sx} vs {sy}")
            fig.update_layout(template=_KANAME_TEMPLATE, height=420, margin=dict(l=20, r=20, t=60, b=20))
            st.plotly_chart(fig, use_container_width=True, key="dash_sc_fig")
            last_figure = fig
        else:
            st.info("Need ≥2 numeric columns for scatter plot. Select more numeric columns.")

    elif chart_type == "Bar":
        if use_cat:
            with p2:
                bc = st.selectbox("Category X", use_cat, index=0, key="dash_bar_x")
                top_n = st.slider("Top N categories", 2, 50, 10, key="dash_bar_top")
            vc = view_df[bc].dropna().value_counts().head(top_n).reset_index()
            vc.columns = [bc, "count"]
            fig = px.bar(vc, x=bc, y="count", color="count",
                         color_continuous_scale=colormap, title=f"Top-{top_n} {bc} counts")
            fig.update_layout(template=_KANAME_TEMPLATE, height=420)
            st.plotly_chart(fig, use_container_width=True, key="dash_bar_fig")
            last_figure = fig
        else:
            st.info("Select a categorical column for a bar chart.")

    elif chart_type == "Box":
        if use_numeric:
            with p2:
                bx = st.selectbox("Numeric Y", use_numeric, index=0, key="dash_box_y")
                bg = st.selectbox("Group by (cat)", ["(none)"] + use_cat, index=0, key="dash_box_g")
                args = {"y": bx}
                if bg != "(none)":
                    args["x"] = bg
            fig = px.box(view_df, **args, title=f"Box plot — {bx}")
            fig.update_layout(template=_KANAME_TEMPLATE, height=420, margin=dict(l=20, r=20, t=60, b=20))
            st.plotly_chart(fig, use_container_width=True, key="dash_box_fig")
            last_figure = fig
        else:
            st.info("Select at least one numeric column for a box plot.")

    elif chart_type == "Violin":
        if use_numeric:
            with p2:
                vy = st.selectbox("Numeric Y", use_numeric, index=0, key="dash_vio_y")
                vg = st.selectbox("Group by (cat)", ["(none)"] + use_cat, index=0, key="dash_vio_g")
                args = {"y": vy, "box": True, "points": "outliers"}
                if vg != "(none)":
                    args["x"] = vg
            fig = px.violin(view_df, **args, title=f"Violin plot — {vy}")
            fig.update_layout(template=_KANAME_TEMPLATE, height=420)
            st.plotly_chart(fig, use_container_width=True, key="dash_vio_fig")
            last_figure = fig
        else:
            st.info("Select at least one numeric column for a violin plot.")

    elif chart_type == "Correlation Heatmap":
        if len(use_numeric) >= 2:
            with p2:
                st.caption(f"Method = Pearson · Colormap = {colormap}")
                zmin = st.slider("z-min", -1.0, 0.0, -1.0, 0.05, key="dash_zmin")
                zmax = st.slider("z-max", 0.0, 1.0, 1.0, 0.05, key="dash_zmax")
            # Prefer backend correlation-matrix endpoint if available
            corr_result = None
            if _REQUESTS_AVAILABLE and file_bytes is not None and filename is not None:
                with st.spinner("Calling /correlation-matrix backend..."):
                    ok_c, rc = _request("POST", "/api/data/correlation-matrix",
                                        files={"file": (filename, file_bytes)},
                                        data={"method": "pearson"},
                                        timeout=60)
                if ok_c and isinstance(rc, dict) and rc.get("correlation"):
                    corr_result = rc["correlation"]
            try:
                if corr_result is not None:
                    z = [[(0.0 if v is None else float(v)) for v in row] for row in corr_result["matrix"]]
                    cols = list(corr_result["columns"])
                else:
                    corr_df = view_df[use_numeric].corr(numeric_only=True)
                    cols = list(corr_df.columns)
                    z = corr_df.values.tolist()
                if cols and z:
                    fig = go.Figure(data=go.Heatmap(
                        z=z, x=cols, y=cols,
                        colorscale=colormap, zmin=zmin, zmax=zmax,
                        hoverongaps=False,
                    ))
                    fig.update_layout(title="Pearson Correlation Heatmap",
                                      template=_KANAME_TEMPLATE, height=480,
                                      margin=dict(l=20, r=20, t=60, b=20),
                                      xaxis_tickangle=-40)
                    st.plotly_chart(fig, use_container_width=True, key="dash_heat_fig")
                    last_figure = fig
            except Exception as e:
                st.error(f"Heatmap error: {e}")
        else:
            st.info("Need ≥2 numeric columns for correlation heatmap.")

    # -------- IQR Outlier Detection --------
    st.markdown("### 6️⃣ IQR Outlier Detection")
    if use_numeric:
        ocol = st.selectbox("Numeric column for outlier scan", use_numeric, index=0, key="dash_outlier_col")
        outlier_result = None
        if _REQUESTS_AVAILABLE and file_bytes is not None and filename is not None:
            with st.spinner("Calling /outliers backend..."):
                ok_o, ro = _request("POST", "/api/data/outliers",
                                    files={"file": (filename, file_bytes)},
                                    data={"column": ocol, "k": str(iqr_k)},
                                    timeout=60)
            if ok_o and isinstance(ro, dict) and ro.get("outliers"):
                outlier_result = ro["outliers"]
        # Fallback local compute
        if outlier_result is None:
            try:
                s = pd.to_numeric(view_df[ocol], errors="coerce").dropna().values
                q1 = float(np.nanpercentile(s, 25))
                q3 = float(np.nanpercentile(s, 75))
                iqr = q3 - q1
                lb = q1 - iqr_k * iqr
                ub = q3 + iqr_k * iqr
                inds = []
                vals = []
                for i, v in enumerate(view_df[ocol].tolist()):
                    try:
                        fv = float(v)
                        if fv < lb or fv > ub:
                            inds.append(i)
                            vals.append(fv)
                    except Exception:
                        continue
                outlier_result = {
                    "column": ocol, "k": float(iqr_k), "q1": q1, "q3": q3,
                    "iqr": iqr, "lower_bound": lb, "upper_bound": ub,
                    "outlier_count": len(inds), "outlier_indices": inds, "outlier_values": vals,
                    "n_total_rows": int(len(view_df)),
                }
            except Exception as e:
                st.caption(f"Local IQR error: {e}")

        if outlier_result:
            oc1, oc2, oc3, oc4, oc5 = st.columns(5)
            oc1.metric("Q1", f"{outlier_result.get('q1', 0):.3f}")
            oc2.metric("Q3", f"{outlier_result.get('q3', 0):.3f}")
            oc3.metric("IQR", f"{outlier_result.get('iqr', 0):.3f}")
            oc4.metric("Lower", f"{outlier_result.get('lower_bound', 0):.3f}")
            oc5.metric("Upper", f"{outlier_result.get('upper_bound', 0):.3f}")
            cnt = int(outlier_result.get("outlier_count") or 0)
            st.metric("Outliers flagged",
                      f"{cnt:,} / {int(outlier_result.get('n_total_rows') or 0):,}",
                      delta=f"{100 * cnt / max(1, int(outlier_result.get('n_total_rows') or 1)):.1f} % of rows")
            if cnt:
                with st.expander(f"Outlier rows — first {min(cnt, 50)}"):
                    oi = list(outlier_result.get("outlier_indices") or [])[:50]
                    ov = list(outlier_result.get("outlier_values") or [])[:50]
                    odf = pd.DataFrame({"row_index": oi, f"{ocol}_value": ov})
                    if len(oi) <= len(view_df):
                        try:
                            extra = view_df.loc[oi].reset_index(drop=True)
                            odf = pd.concat([odf, extra], axis=1)
                        except Exception:
                            pass
                    st.dataframe(odf, use_container_width=True)
    else:
        st.info("Select at least one numeric column to scan for outliers.")

    # -------- WIDGET 8: Download CSV + PNG --------
    st.markdown("")
    dl1, dl2, dl3 = st.columns([1, 1, 2])
    with dl1:
        csv_buf = io.BytesIO()
        view_df.to_csv(csv_buf, index=False, encoding="utf-8-sig")
        csv_bytes = csv_buf.getvalue()
        st.download_button(
            "⬇️ Download (CSV)",
            data=csv_bytes,
            file_name=("filtered_" + (filename or "dataset.csv").rsplit(".", 1)[0] + ".csv"),
            mime="text/csv",
            key="dash_dl_csv",
        )
    with dl2:
        png_bytes = None
        if last_figure is not None:
            try:
                png_bytes = last_figure.to_image(format="png", engine="kaleido", scale=2)
            except Exception:
                try:
                    png_bytes = last_figure.to_image(format="png", scale=2)
                except Exception:
                    png_bytes = None
        if png_bytes:
            st.download_button(
                "⬇️ Download (PNG)",
                data=png_bytes,
                file_name=(("chart_" + chart_type.replace(" ", "_") + "_" + (filename or "figure").rsplit(".", 1)[0]) + ".png"),
                mime="image/png",
                key="dash_dl_png",
            )
        else:
            st.caption("PNG requires an active figure above + kaleido/orca install.")
    with dl3:
        with st.expander("Raw analysis JSON (last backend /analyze response)"):
            if file_bytes and filename and _REQUESTS_AVAILABLE:
                with st.spinner("Fetching raw JSON..."):
                    ok_r, rp = _request("POST", f"/api/data/analyze?sample_rows={sample_rows}",
                                        files={"file": (filename, file_bytes)},
                                        data={"sample_rows": str(sample_rows)},
                                        timeout=120)
                if ok_r and isinstance(rp, dict):
                    st.json(rp)
                else:
                    st.caption(str(rp) if isinstance(rp, str) else "Backend unavailable")
            else:
                st.caption("Upload a dataset first to view raw JSON.")


# ---------------------------------------------------------------------------
# MODULE: TRANSLATION
# ---------------------------------------------------------------------------

TRANSLATION_SAMPLES = {
    "JA → EN ビジネス文": "御社の担当者様におかれましては、平素より格別のご高配を賜り、厚く御礼申し上げます。",
    "JA → EN 技術文": "Pythonで構築した機械学習モデルをGPUクラスタ上で学習させ、推論APIとして公開しました。",
    "EN → JA Business": "Thank you for your detailed email. Our team will review the proposal and contact your manager by EOD tomorrow.",
    "EN → JA Technical": "We deployed the microservices to the Kubernetes cluster and configured horizontal pod autoscaling based on CPU utilization.",
}

GLOSSARY_DOMAINS = {
    "business_general": "一般ビジネス (Business General)",
    "technology_ai": "IT・AI技術 (Technology / AI)",
    "finance_accounting": "経理・財務 (Finance / Accounting)",
    "legal_contract": "契約・法務 (Legal / Contract)",
}


def render_translation():
    st.markdown("## Translation Intelligence (JP ↔ EN)")
    st.markdown('<span class="status-badge status-impl">Implemented — LLM Hybrid (Claude / Gemini) + Glossary + Dictionary Fallback</span>',
                unsafe_allow_html=True)
    st.caption("🌐 Scope: KANAME translates strictly between **Japanese and English (JA ↔ EN)** only. "
               "The Document module reuses the same JA↔EN engine to auto-translate English input to Japanese before generation.")
    st.markdown("")

    engine_info = None
    if _REQUESTS_AVAILABLE:
        ok_t, msg = _request("GET", "/api/translation/health")
        if ok_t and isinstance(msg, dict):
            engine_info = msg
            st.success(
                f"Active engine: **{msg.get('active_engine', '?')}**  ·  "
                f"Configured LLM keys: {len(msg.get('configured_keys', [])) or 'fallback only'}  ·  "
                f"Domains: {', '.join(msg.get('domains') or [])}"
            )
            if msg.get("active_engine") == "dictionary":
                st.info("Fallback dictionary mode active. Add ANTHROPIC_API_KEY / GOOGLE_API_KEY to `.env` for LLM-quality output.")
        else:
            st.error(msg if isinstance(msg, str) else "Backend offline")

    with st.expander("About this module", expanded=False):
        st.markdown("""
            **Engine Pipeline:**
            1. Glossary pre-processing (4 bilingual domain tables × ~30 pairs each)
            2. LLM backend auto-detection: **Claude Haiku → Gemini 2.0 Flash**
            3. Dictionary fallback if no LLM key configured or LLM fails
            - Quality heuristic score (0–1): length ratio, numeric preservation, glossary coverage, format preservation
        """)

    st.markdown("")
    tab_main, tab_glossary, tab_eval = st.tabs(["⚡ Translate", "📚 Domain Glossaries", "📊 Quality Evaluation"])

    with tab_main:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            sample_key = st.selectbox("Sample", options=list(TRANSLATION_SAMPLES.keys()), index=0, key="ts_samp")
        with c2:
            src_lang = st.selectbox("Source", ["auto", "ja", "en"], index=0, format_func=lambda x: {"auto": "Auto-detect", "ja": "🇯🇵 Japanese", "en": "🇺🇸 English"}[x], key="ts_src")
        with c3:
            tgt_lang = st.selectbox("Target", ["en", "ja"], index=0, format_func=lambda x: {"ja": "🇯🇵 Japanese", "en": "🇺🇸 English"}[x], key="ts_tgt")
        with c4:
            dom_list = ["(none)"] + list(GLOSSARY_DOMAINS.keys())
            domain_raw = st.selectbox("Domain hint", dom_list, index=0, format_func=lambda x: GLOSSARY_DOMAINS.get(x, "No specific domain") if x != "(none)" else "No specific domain", key="ts_dom")
            domain_val = None if domain_raw == "(none)" else domain_raw

        default_text = TRANSLATION_SAMPLES.get(sample_key, "")
        col_s, col_t = st.columns(2)
        with col_s:
            text = st.text_area("Source text", value=default_text, height=180, key="ts_src_txt",
                                placeholder="Enter Japanese or English text here...")
        with col_t:
            translated_out = st.empty()
            out_metrics = st.container()

        with st.expander("Custom terminology + few-shot style examples", expanded=False):
            st.caption("JSON arrays are applied per request. Examples guide an available LLM; they do not fine-tune model weights.")
            custom_glossary_raw = st.text_area(
                "Custom glossary [{\"ja\":\"用語\",\"en\":\"term\"}]",
                value="[]",
                key="translation_custom_glossary",
            )
            translation_examples_raw = st.text_area(
                "Reference examples [{\"source\":\"...\",\"target\":\"...\"}]",
                value="[]",
                key="translation_examples",
            )

        run = st.button("🌐 Translate", type="primary", use_container_width=True, key="ts_run")

        if run and text.strip() and _REQUESTS_AVAILABLE:
            try:
                custom_glossary = json.loads(custom_glossary_raw)
                examples = json.loads(translation_examples_raw)
                if not isinstance(custom_glossary, list) or not isinstance(examples, list):
                    raise ValueError("Glossary and examples must each be JSON arrays.")
                with st.spinner("Calling KANAME translation backend..."):
                    ok, resp = _request(
                        "POST", "/api/translation/translate",
                        payload={
                            "text": text, "source_lang": src_lang, "target_lang": tgt_lang,
                            "domain": domain_val, "custom_glossary": custom_glossary,
                            "examples": examples,
                        },
                        timeout=180,
                    )
            except (ValueError, json.JSONDecodeError) as exc:
                ok, resp = False, f"Invalid glossary/example JSON: {exc}"
            if ok and isinstance(resp, dict):
                with col_t:
                    translated_out.text_area("Translated text", value=resp.get("translated_text", ""), height=180, key="ts_out_txt")
                    qual = resp.get("quality") or {}
                    mk1, mk2, mk3, mk4, mk5 = out_metrics.columns(5)
                    mk1.metric("Overall", f"{round(float(qual.get('overall',0)),2):.2f}")
                    mk2.metric("Length", f"{round(float(qual.get('length_ratio_score',0)),2):.2f}")
                    mk3.metric("Numbers", f"{round(float(qual.get('numeric_preservation_score',0)),2):.2f}")
                    mk4.metric("Glossary", f"{round(float(qual.get('glossary_coverage_score',0)),2):.2f}")
                    mk5.metric("Format", f"{round(float(qual.get('format_preservation_score',0)),2):.2f}")

                    meta_cols = st.columns(4)
                    meta_cols[0].caption(f"Engine: `{resp.get('engine','?')}`")
                    meta_cols[1].caption(f"Detected source: `{resp.get('source_lang','?')}`")
                    meta_cols[2].caption(f"Glossary hits: {resp.get('glossary_hits', 0)}")
                    meta_cols[3].caption(f"Domain: {resp.get('domain') or '(default)'}")

                with st.expander("Raw JSON response + quality breakdown"):
                    st.json(resp)
            elif not ok:
                st.error(f"Translation failed: {resp}")
        elif run and not text.strip():
            st.warning("Enter some text to translate.")

    with tab_glossary:
        sel_domain = st.selectbox("Select domain", list(GLOSSARY_DOMAINS.keys()), format_func=lambda x: GLOSSARY_DOMAINS[x], key="tg_dom")
        if _REQUESTS_AVAILABLE:
            ok_g, gresp = _request("GET", f"/api/translation/glossaries/{sel_domain}")
            if ok_g and isinstance(gresp, dict) and gresp.get("pairs"):
                pairs = gresp["pairs"]
                st.markdown(f"**{len(pairs)} pairs · {GLOSSARY_DOMAINS[sel_domain]}**")
                df_gloss = pd.DataFrame(pairs)
                st.dataframe(df_gloss, use_container_width=True, hide_index=True, height=520)
            elif ok_g:
                st.caption("No pairs for this domain.")
            else:
                st.caption(str(gresp))

    with tab_eval:
        st.markdown("**Offline evaluation (deterministic dictionary pipeline, no LLM keys or network required)**")
        if not _REQUESTS_AVAILABLE:
            st.caption("`requests` package not available — cannot reach backend.")
        else:
            ok_e, eval_resp = _request("GET", "/api/translation/evaluate", timeout=120)
            if ok_e and isinstance(eval_resp, dict):
                by = eval_resp.get("by_direction", {})
                for direction, stats in by.items():
                    st.markdown(f"**{direction.upper()}**")
                    m1, m2, m3 = st.columns(3)
                    m1.metric("BLEU (smoothed)", f"{float(stats.get('bleu', 0)):.3f}")
                    m2.metric("Char-F1", f"{float(stats.get('avg_char_f1', 0)):.3f}")
                    m3.metric("Heuristic quality", f"{float(stats.get('avg_quality', 0)):.3f}")
                    prec = stats.get("bleu_precisions") or []
                    if prec:
                        st.caption(f"BLEU precisions (1–4-gram): {', '.join(f'{p:.3f}' for p in prec)}")
                overall = eval_resp.get("overall", {})
                st.divider()
                o1, o2 = st.columns(2)
                o1.metric("Overall char-F1", f"{float(overall.get('avg_char_f1', 0)):.3f}")
                o2.metric("Overall quality", f"{float(overall.get('avg_quality', 0)):.3f}")

                ok_r, rows_resp = _request("GET", "/api/translation/evaluate?include_rows=true", timeout=120)
                if ok_r and isinstance(rows_resp, dict) and rows_resp.get("rows"):
                    st.markdown("##### Per-sentence comparison")
                    st.dataframe(pd.DataFrame(rows_resp["rows"]), use_container_width=True, hide_index=True)
            else:
                st.error(eval_resp if isinstance(eval_resp, str) else "Evaluation unavailable")


# ---------------------------------------------------------------------------
# MODULE: DOCUMENT INTELLIGENCE
# ---------------------------------------------------------------------------

OFFICIAL_TEMPLATE_SAMPLES = {
    "japanese-resume": {
        "full_name_kanji": "山田 太郎", "full_name_kana": "やまだ たろう", "gender": "male",
        "date_of_birth": "1990-01-01", "phone": "090-1234-5678", "email": "taro@example.com",
        "postal_code": "100-0001", "address_prefecture": "東京都", "address_city": "千代田区",
        "address_detail": "大手町1-1-1 KANAMEビル301",
        "education_list": [
            {"start_date": "2008-04", "end_date": "2012-03", "school_name": "東京大学", "department": "工学部", "degree": "学士（工学）", "is_graduated": True}
        ],
        "work_history_list": [
            {"start_date": "2012-04", "end_date": None, "company_name": "KANAME株式会社", "position": "シニアエンジニア", "description": "AIプロダクト開発", "is_current": True}
        ],
        "licenses_list": [{"acquisition_date": "2020-06", "license_name": "技術士（情報工学）", "issuer": "技術士会"}],
        "self_pr": "Python・機械学習・チームリーディングを強みとしています。大型プロジェクトを複数リードし、QCD改善に貢献してきました。",
        "motivation": "貴社のAI事業拡大に技術面から貢献したく、応募いたしました。",
    },
    "career-history": {
        "full_name_kanji": "山田 太郎", "full_name_kana": "ヤマダ タロウ",
        "phone": "090-1234-5678", "email": "taro@example.com",
        "career_summary": "Pythonバックエンドエンジニアとして5年、AI/MLプロダクトの開発リードを3年経験。チーム10名のマネジメント実績あり。",
        "roles": [
            {"period_from": "2022-04", "period_to": None, "company_name": "KANAME株式会社", "position": "Engineering Manager / Senior Engineer", "department": "AI事業部", "overview": "LLM基盤・音声認識製品の開発リード、アジャイルチームのマネジメント"},
            {"period_from": "2019-04", "period_to": "2022-03", "company_name": "テスト商事株式会社", "position": "Software Engineer", "department": "基盤開発部", "overview": "決済API・データ基盤の開発・運用"},
        ],
        "skills": ["Python", "FastAPI", "PostgreSQL", "scikit-learn", "AWS", "Docker", "Kubernetes"],
        "education": "東京大学 工学部 電子情報工学科 卒業",
        "certifications": ["技術士（情報工学部門）", "AWS Certified Solutions Architect – Associate"],
    },
    "invoice": {
        "invoice_number": "INV-2026-0123", "issue_date": "2026-03-31", "due_date": "2026-04-30",
        "seller_name": "KANAME株式会社", "seller_department": "開発部",
        "seller_address": "東京都千代田区大手町1-1-1 KANAMEビル",
        "seller_phone": "03-1234-5678", "seller_tax_id": "T-0000000000000",
        "buyer_name": "テスト商事株式会社", "buyer_department": "購買部",
        "buyer_address": "東京都渋谷区渋谷1-2-3 テストビル",
        "title": "AIシステム構築業務 請求書",
        "items": [
            {"description": "コンサルティング業務（50h × 2名）", "quantity": 100, "unit_price": 12000, "tax_rate": 0.10},
            {"description": "LLM基盤 開発業務", "quantity": 1, "unit_price": 1800000, "tax_rate": 0.10},
        ],
        "payment_method": "bank_transfer",
        "bank_name": "みずほ銀行", "bank_branch": "大手町支店",
        "bank_account_type": "ordinary", "bank_account_number": "1234567",
        "bank_account_holder": "カ）カナメ",
        "notes": "お支払期限を過ぎた場合、所定の遅延損害金を申し受ける場合がございます。",
    },
    "business-report": {
        "title": "2026年度 第1四半期 業務報告書",
        "subtitle": "AI事業部 開発チーム",
        "author": "山田 太郎", "author_department": "AI事業部 開発部",
        "report_date": "2026-03-31",
        "recipient": "鈴木 事業部長", "recipient_department": "AI事業部",
        "executive_summary": "第1四半期は新規AIプロダクト2件をリリースし、売上目標120%を達成した。",
        "background": "2025年度末より進めていたLLM基盤を本番リリースし、3事業部へ横展開を開始した。",
        "objectives": ["売上高 前年比 +15% 達成", "新規製品 2件 リリース", "稼働率 99.9% 維持"],
        "methods": "アジャイル開発（2週間スプリント×6回）、CI/CDパイプライン強化、SLOベースの運用監視。",
        "results": "売上高は前年同期比 +22% となり、目標を大幅に上回った。新規顧客は12社獲得。",
        "discussion": "要因は①既存製品の顧客満足度向上 ②LLM基盤の採用加速 ③販売チームとの連携強化 である。",
        "issues": ["データ基盤のパフォーマンスチューニング（Q2対応予定）", "技術的負債の解消ペース改善が必要"],
        "conclusion": "第1四半期は総じて良好な結果であり、通期計画に対して前倒しで進捗している。",
        "recommendations": ["Q2はデータ基盤のアーキテクチャ刷新を推進", "新規市場向けPoCを3件 開始"],
        "references": ["- KANAME内部文書: 2026年度事業計画書 v1.3"],
        "attachments": ["- 別紙1: 財務サマリー", "- 別紙2: チーム別KPI一覧"],
    },
    "cover-letter": {
        "sender_company": "KANAME株式会社", "sender_department": "開発部",
        "sender_name": "山田 太郎", "sender_postal_code": "100-0001",
        "sender_address": "東京都千代田区大手町1-1-1 KANAMEビル",
        "sender_phone": "03-1234-5678", "sender_email": "taro.yamada@kaname.co.jp",
        "recipient_company": "テスト商事株式会社", "recipient_department": "人事部 採用担当",
        "recipient_title": "ご担当者様", "recipient_name": "御中",
        "letter_date": "2026-03-31",
        "subject": "応募書類送付のご案内",
        "greeting": "お世話になっております。",
        "body_paragraphs": [
            "このたび、貴社求人募集に応募させていただきたく、下記の通り書類を送付いたします。",
            "ご多忙のところ恐縮ではございますが、ご審査のほど何卒よろしくお願い申し上げます。",
            "まずは、書類送付のご挨拶まで申し上げます。",
        ],
        "closing": "今後とも何卒よろしくお願い申し上げます。",
        "enclosures_title": "添付書類一覧",
        "enclosures": ["履歴書（原本）×1通", "職務経歴書×1通", "卒業証明書の写し×1通", "職務経歴に関する参考資料×2通"],
    },
    "meeting-minutes": {
        "title": "2026年度 第1回 定例プロジェクト会議 議事録",
        "meeting_date": "2026-04-10", "start_time": "14:00", "end_time": "15:30",
        "location": "本社 3階 会議室A", "meeting_type": "regular",
        "chairperson": "鈴木一郎", "recorder": "佐藤花子",
        "attendees": ["鈴木一郎", "佐藤花子", "山田太郎", "田中次郎", "高橋三郎"],
        "absentees": ["伊藤四郎"],
        "agenda": ["前回議事録の確認", "第1四半期の進捗報告", "文書自動生成システムのリリース判定", "次期開発計画の協議"],
        "discussion": "前回議事録の内容を確認し、異議なく承認。第1四半期の進捗は売上・品質ともに目標超過。文書自動生成システムは生成成功率99.6%を達成し、リリース基準を満たすことを確認した。次期開発は音声認識の横展開を優先する方針で合意した。",
        "decisions": ["文書自動生成システムを4月15日に本番リリースする。", "次期開発は音声認識の横展開を最優先とする。"],
        "action_items": [
            {"task": "リリース判定資料の最終化", "owner": "佐藤花子", "due_date": "2026-04-12"},
            {"task": "本番環境のスモークテスト実施", "owner": "山田太郎", "due_date": "2026-04-14"},
        ],
        "next_meeting_date": "2026-05-08",
    },
    "custom-document": {
        "title": "新規システム導入 ご提案書",
        "author": "佐藤健一", "document_date": "2026-04-15",
        "recipient": "株式会社ABC商事 御中",
        "intro": "平素より大変お世話になっております。この度、貴社の業務効率化を目的とした新規システム導入について、下記のとおりご提案申し上げます。",
        "fields": [
            {"label": "案件名", "value": "KANAME 文書自動生成システム導入"},
            {"label": "提案金額", "value": "3,500,000円（税別）"},
            {"label": "導入予定期間", "value": "2026年5月 ～ 2026年7月"},
        ],
        "sections": [
            {"heading": "1. 提案の背景", "body": "貴社では定型文書の作成業務が多く、月間で約120時間の工数が発生している。"},
            {"heading": "2. 提案内容", "body": "KANAME の文書自動生成システムを導入し、定型文書をテンプレート化する。API連携も可能。"},
            {"heading": "3. 期待される効果", "body": "文書作成工数を約30%削減し、表記ゆれの解消と監査対応の効率化を実現する。"},
        ],
        "closing": "ご検討の程、何卒よろしくお願い申し上げます。",
    },
}

TEMPLATE_LABELS = {
    "japanese-resume": ("履歴書 Rirekisho", "Standard Japanese resume with photo box, education/work, licenses, self PR."),
    "career-history": ("職務経歴書 Career History", "Detailed career summary, roles matrix, skill stack, certifications."),
    "invoice": ("請求書 Seikyu-Sho (Invoice)", "Qualified invoice with tax, T-number, bank info, items, subtotal/total."),
    "business-report": ("一般報告書 Business Report", "11-chapter structured corporate report: summary, background, objectives, methods, results..."),
    "cover-letter": ("送付状 Soufu-Jo (Cover Letter)", "Standard Japanese transmittal sheet with sender/recipient blocks and numbered enclosures."),
    "meeting-minutes": ("議事録 Meeting Minutes", "Meeting minutes with date/time, attendees, agenda, discussion, decisions, action items."),
    "custom-document": ("自由書式文書 Custom Document", "Free-format document: user-defined sections (heading + body) plus custom key-value fields."),
}


def render_document():
    st.markdown("## Document Intelligence")
    st.markdown('<span class="status-badge status-impl">Implemented — python-docx + ReportLab (JP CID font) · 5 Official Japanese Templates</span>',
                unsafe_allow_html=True)
    st.markdown("")

    if _REQUESTS_AVAILABLE:
        ok, msg = _request("GET", "/api/documents/health")
        if ok and isinstance(msg, dict):
            st.success(
                f"Engine: {msg.get('engine')}  ·  "
                f"Formats: {', '.join(msg.get('supported_formats') or [])}  ·  "
                f"Generators: {', '.join(msg.get('available_generators') or [])}"
            )
        else:
            st.error(msg if isinstance(msg, str) else "Backend offline")

    auto_translate_docs = st.checkbox(
        "🌐 Auto-translate English input → Japanese before generation (KANAME supports JA↔EN only)",
        value=True,
        key="doc_auto_translate",
        help="When enabled, English free-text fields are translated to Japanese first and then the "
             "document is generated. Disable it to use the text exactly as entered (e.g. you already "
             "typed Japanese). Proper-noun fields such as names are always kept as entered.",
    )

    with st.expander("Module Description", expanded=False):
        st.markdown("""
            **Active generators (2 categories, 6 total):**
            1. `nlp-report` — Bilingual DOCX/PDF report of Japanese morphological analysis
               - DOCX: python-docx with styled sections, bilingual headings
               - PDF: ReportLab with `HeiseiKakuGo-W5` CID Japanese font
            2. Official Japanese templates × 5 × (DOCX / PDF):
               - 履歴書 Rirekisho, 職務経歴書, 請求書, 一般報告書, 送付状
               - All templates accept JSON field data via the FastAPI endpoint
               - Downloadable binary files with RFC 5987 UTF-8 filename headers
        """)

    tab_nlp, tab_resume, tab_career, tab_invoice, tab_report, tab_cover, tab_expert, tab_recent, tab_tools = st.tabs(
        ["🧾 NLP Report", "📄 履歴書 Resume", "📑 職務経歴書 Career", "💴 請求書 Invoice", "📘 報告書 Report", "✉ 送付状 Cover", "🧪 Expert JSON", "📂 Recent Docs", "🧰 Batch + AI Tools"]
    )

    # Shared helper for generate + download (so each tab calls DRY)
    def _doc_generate_and_download(tmpl_id: str, payload_obj: Dict[str, Any], fmt: str,
                                   auto_translate: bool = True) -> None:
        if not _REQUESTS_AVAILABLE:
            st.error("requests not installed")
            return
        at_params = {"auto_translate": "true" if auto_translate else "false"}
        with st.spinner(f"Generating {tmpl_id} · {fmt.upper()} via backend..."):
            ok_meta, meta = _request(
                "POST", f"/api/documents/generate/{tmpl_id}",
                payload=payload_obj, timeout=120, params=at_params,
            )
        meta_c1, meta_c2, meta_c3, meta_c4 = st.columns(4)
        if ok_meta and isinstance(meta, dict) and not meta.get("error"):
            meta_c1.metric("Template", tmpl_id)
            meta_c2.metric("Format", fmt.upper())
            meta_c3.metric("Size (bytes)", f"{int(meta.get('size_bytes') or 0):,}")
            meta_c4.metric("Ledger ID", (meta.get("document_id") or "—")[:8] + "…" if meta.get("document_id") else "—")
            tr = meta.get("auto_translate")
            if isinstance(tr, dict) and tr.get("translated_count"):
                st.info(
                    f"🌐 Auto-translated {tr.get('translated_count')} English field(s) → "
                    f"{str(tr.get('target_lang', 'ja')).upper()} before generation."
                )
                with st.expander("Translated fields (English → Japanese)", expanded=False):
                    st.dataframe(pd.DataFrame(tr.get("translated_fields") or []),
                                 use_container_width=True, hide_index=True)
            with st.expander("Ledger metadata (full)", expanded=False):
                st.json(meta)
        else:
            if ok_meta and isinstance(meta, dict):
                st.error("Metadata error: " + str(meta.get("error", meta)))
            else:
                st.warning("Metadata failed: " + str(meta))
        with st.spinner(f"Preparing {fmt.upper()} download..."):
            ok_dl, resp = _request(
                "POST", f"/api/documents/generate/{tmpl_id}/download",
                payload=payload_obj, timeout=180, raw=True, params=at_params,
            )
        if ok_dl and hasattr(resp, "content"):
            content_type = resp.headers.get("Content-Type", "")
            disp = resp.headers.get("Content-Disposition", "")
            fname = _extract_filename_from_disp(disp, f"{tmpl_id}_output.{fmt}")
            bytes_data = resp.content
            st.success(f"✅ Ready · {len(bytes_data):,} bytes.")
            st.download_button(
                label=f"⬇ Download {fname}",
                data=bytes_data,
                file_name=fname,
                mime=content_type or "application/octet-stream",
                type="primary",
                use_container_width=True,
                key=f"dl_{tmpl_id}_{fmt}_{abs(hash(json.dumps(payload_obj, ensure_ascii=False, sort_keys=True))) % 1000000}",
            )
        else:
            st.error(f"Download generation failed: {resp}")

    with tab_nlp:
        st.markdown("### 🧾 Generate: NLP Bilingual Report")
        with st.form("doc_form"):
            r1, r2 = st.columns([3, 2])
            with r1:
                title = st.text_input("Report title", value="KANAME NLP Analysis Report")
                author = st.text_input("Author", value="KANAME [要]")
                sample = st.selectbox("Use sample text", options=list(SAMPLE_TEXTS.keys()), index=2)
                text = st.text_area("Japanese text for analysis", value=SAMPLE_TEXTS[sample],
                                    height=160, key="doc_text")
            with r2:
                format_opt = st.selectbox("Format", ["docx", "pdf"], index=0)
                mode = st.selectbox("Sudachi Mode", ["C", "B", "A"], index=0)
                bilingual = st.checkbox("Include bilingual notes section", value=True)
            generate = st.form_submit_button("📄 Generate Report", type="primary", use_container_width=True)

        if generate and text.strip():
            payload = {
                "text": text, "mode": mode, "title": title,
                "include_bilingual": bilingual, "doc_format": format_opt, "author": author,
            }
            _doc_generate_and_download("nlp-report", payload, format_opt, auto_translate_docs)

    # ---------------- TAB: 履歴書 (Resume) ----------------
    with tab_resume:
        st.markdown("### 📄 履歴書 (Japanese Resume — Rirekisho)")
        t_default = OFFICIAL_TEMPLATE_SAMPLES["japanese-resume"]
        with st.form("rirekisho_form"):
            rc1, rc2 = st.columns(2)
            with rc1:
                full_name_kanji = st.text_input("氏名 (漢字 / Full name Kanji)", t_default.get("full_name_kanji", "山田 太郎"))
                full_name_kana = st.text_input("氏名 (ふりがな / Reading)", t_default.get("full_name_kana", "やまだ たろう"))
                gender = st.selectbox("性別 / Gender", ["male", "female", "other"], index=0 if t_default.get("gender","male")=="male" else 1)
                dob = st.date_input("生年月日 / Date of birth",
                                    value=datetime.strptime(t_default.get("date_of_birth","1990-01-01"), "%Y-%m-%d").date())
                age = st.number_input("年齢 / Age", 0, 120, int(t_default.get("age", 35)))
                postal_code = st.text_input("郵便番号 / Postal code", t_default.get("postal_code", "100-0001"))
                address = st.text_area("住所 / Address", t_default.get("address", "東京都千代田区大手町1-1-1"), height=80)
                phone = st.text_input("電話 / Phone", t_default.get("phone", "03-1234-5678"))
                email = st.text_input("メール / Email", t_default.get("email", "taro.yamada@kaname.co.jp"))
            with rc2:
                nearest_station = st.text_input("最寄駅 / Nearest station", t_default.get("nearest_station", "大手町駅"))
                marital_status = st.selectbox("婚姻 / Marital", ["married", "unmarried"],
                                              index=0 if t_default.get("marital_status","married")=="married" else 1)
                has_driver_license = st.checkbox("普通自動車免許 / Driver license",
                                                 value=bool(t_default.get("has_driver_license")))
                photo_path = st.text_input("写真パス / Photo path (optional)", t_default.get("photo_path",""))
                desired_salary = st.text_input("希望年収 / Desired salary", t_default.get("desired_salary","年収 600万円"))
                pr_title = st.text_input("自己PR タイトル / Self PR title",
                                         t_default.get("self_pr_title", "LLM基盤開発とチームビルディング"))
                pr_body = st.text_area("自己PR / Self PR",
                                       t_default.get("self_pr_body",
                                                     "深層学習基盤の設計・開発・運用に7年間従事。5~12名のチームリーダー経験3回。"),
                                       height=120)
            st.markdown("**学歴 / Education (expand to edit rows)**")
            edu_rows = list(t_default.get("education", [])) or []
            edu_df = pd.DataFrame(edu_rows) if edu_rows else pd.DataFrame([{"period": "2008年4月", "detail": "○○県立○○高等学校 入学"}])
            edu_edited = st.data_editor(edu_df, num_rows="dynamic", use_container_width=True, key="edu_df_edit", hide_index=True)
            st.markdown("**職歴 / Work history**")
            work_rows = list(t_default.get("work_history", [])) or []
            work_df = pd.DataFrame(work_rows) if work_rows else pd.DataFrame([{"period":"2015年4月","detail":"○○株式会社 入社"}])
            work_edited = st.data_editor(work_df, num_rows="dynamic", use_container_width=True, key="work_df_edit", hide_index=True)
            st.markdown("**資格・免許 / Licenses & certifications**")
            lic_rows = list(t_default.get("licenses", [])) or []
            lic_df = pd.DataFrame(lic_rows) if lic_rows else pd.DataFrame([{"period": "2018年7月", "detail": "普通自動車第一種運転免許 取得"}])
            lic_edited = st.data_editor(lic_df, num_rows="dynamic", use_container_width=True, key="lic_df_edit", hide_index=True)
            hobbies = st.text_area("趣味・特技 / Hobbies, skills",
                                   t_default.get("hobbies_skills", "読書・ジョギング・日本語教育"), height=60)
            resume_fmt = st.selectbox("Format", ["docx", "pdf"], index=0, key="resume_fmt")
            gen_resume = st.form_submit_button("📄 Generate 履歴書", type="primary", use_container_width=True)
        if gen_resume:
            payload = {
                "full_name_kanji": full_name_kanji, "full_name_kana": full_name_kana,
                "gender": gender, "date_of_birth": dob.isoformat(), "age": int(age),
                "postal_code": postal_code, "address": address, "phone": phone,
                "email": email, "nearest_station": nearest_station,
                "marital_status": marital_status, "has_driver_license": bool(has_driver_license),
                "photo_path": photo_path or None,
                "education": edu_edited.to_dict(orient="records"),
                "work_history": work_edited.to_dict(orient="records"),
                "licenses": lic_edited.to_dict(orient="records"),
                "self_pr_title": pr_title, "self_pr_body": pr_body,
                "hobbies_skills": hobbies,
                "desired_salary": desired_salary,
                "doc_format": resume_fmt,
            }
            _doc_generate_and_download("japanese-resume", payload, resume_fmt, auto_translate_docs)

    # ---------------- TAB: 職務経歴書 (Career History) ----------------
    with tab_career:
        st.markdown("### 📑 職務経歴書 (Career History — Shokumu Keirekisho)")
        c_default = OFFICIAL_TEMPLATE_SAMPLES["career-history"]
        with st.form("career_form"):
            cc1, cc2 = st.columns(2)
            with cc1:
                c_name = st.text_input("氏名 / Name", c_default.get("name", "山田 太郎"))
                c_title = st.text_input("肩書き / Title", c_default.get("title", "AI基盤エンジニア / テックリード"))
                c_email = st.text_input("Email", c_default.get("email", "taro.yamada@kaname.co.jp"))
                c_phone = st.text_input("Phone", c_default.get("phone", "03-1234-5678"))
            with cc2:
                c_total_exp = st.number_input("総経験年数 / Total years", 0, 60, int(c_default.get("total_experience_years", 10)))
                c_summary = st.text_area("職務要約 / Executive summary",
                                         c_default.get("executive_summary",
                                                       "LLM基盤・機械学習パイプラインの設計から運用まで一貫して担当。"),
                                         height=110)
            st.markdown("**Roles matrix (職務内容マトリクス)**")
            roles_rows = list(c_default.get("roles_matrix") or [])
            roles_df = pd.DataFrame(roles_rows) if roles_rows else pd.DataFrame(columns=["period","company","position","summary"])
            roles_edited = st.data_editor(roles_df, num_rows="dynamic", key="roles_edit", hide_index=True, use_container_width=True)
            st.markdown("**Skill stack (スキルスタック)**")
            sk_rows = list(c_default.get("skill_stack") or [])
            sk_df = pd.DataFrame(sk_rows) if sk_rows else pd.DataFrame(columns=["category","items","level"])
            sk_edited = st.data_editor(sk_df, num_rows="dynamic", key="sk_edit", hide_index=True, use_container_width=True)
            st.markdown("**Key achievements (業務実績・成果)**")
            ka_rows = list(c_default.get("key_achievements") or [])
            ka_df = pd.DataFrame(ka_rows) if ka_rows else pd.DataFrame(columns=["year","project","scale","impact"])
            ka_edited = st.data_editor(ka_df, num_rows="dynamic", key="ka_edit", hide_index=True, use_container_width=True)
            cert_rows = list(c_default.get("certifications") or [])
            st.markdown("**Certifications (資格・検定)**")
            cert_df = pd.DataFrame(cert_rows) if cert_rows else pd.DataFrame(columns=["name","year","issuer"])
            cert_edited = st.data_editor(cert_df, num_rows="dynamic", key="cert_edit", hide_index=True, use_container_width=True)
            career_fmt = st.selectbox("Format", ["docx", "pdf"], index=0, key="career_fmt")
            gen_career = st.form_submit_button("📄 Generate 職務経歴書", type="primary", use_container_width=True)
        if gen_career:
            payload = {
                "name": c_name, "title": c_title, "email": c_email, "phone": c_phone,
                "total_experience_years": int(c_total_exp), "executive_summary": c_summary,
                "roles_matrix": roles_edited.to_dict(orient="records"),
                "skill_stack": sk_edited.to_dict(orient="records"),
                "key_achievements": ka_edited.to_dict(orient="records"),
                "certifications": cert_edited.to_dict(orient="records"),
                "doc_format": career_fmt,
            }
            _doc_generate_and_download("career-history", payload, career_fmt, auto_translate_docs)

    # ---------------- TAB: 請求書 (Invoice) ----------------
    with tab_invoice:
        st.markdown("### 💴 請求書 (Invoice — Seikyu-Sho)")
        i_default = OFFICIAL_TEMPLATE_SAMPLES["invoice"]
        with st.form("invoice_form"):
            ic1, ic2 = st.columns(2)
            with ic1:
                inv_no = st.text_input("請求書番号 / Invoice #", i_default.get("invoice_number","INV-2026-0042"))
                inv_date = st.date_input("請求日 / Issue date",
                                         datetime.strptime(i_default.get("issue_date","2026-03-31"),"%Y-%m-%d").date())
                due_date = st.date_input("支払期日 / Due date",
                                         datetime.strptime(i_default.get("due_date","2026-04-30"),"%Y-%m-%d").date())
                seller = st.text_input("売り手 / Seller (会社名)", i_default.get("seller_name","KANAME株式会社"))
                seller_addr = st.text_area("売り手住所 / Seller address", i_default.get("seller_address","東京都千代田区大手町1-1-1"), height=60)
                seller_tel = st.text_input("売り手TEL", i_default.get("seller_phone","03-1234-5678"))
                seller_tax = st.text_input("T番号 / Seller tax number", i_default.get("seller_t_number","T1234567890123"))
            with ic2:
                buyer = st.text_input("買い手 / Buyer (会社名)", i_default.get("buyer_name","テスト商事株式会社"))
                buyer_addr = st.text_area("買い手住所 / Buyer address", i_default.get("buyer_address","神奈川県横浜市西区みなとみらい2-2-2"), height=60)
                buyer_dep = st.text_input("買い手部署 / Buyer dept", i_default.get("buyer_department","購買部"))
                title = st.text_input("表題 / Title", i_default.get("title","3月分 業務委託料 請求書"))
                subtitle = st.text_input("備考 / Subtitle", i_default.get("subtitle",""))
                default_tax = st.number_input("消費税率 / Default tax rate", 0.0, 0.30, float(i_default.get("default_tax_rate", 0.10)), step=0.01)
            st.markdown("**明細 / Items**")
            items_rows = list(i_default.get("items", [])) or []
            if items_rows:
                items_df = pd.DataFrame(items_rows)
            else:
                items_df = pd.DataFrame([{"description":"サンプル商品","quantity":1,"unit_price":0,"tax_rate":0.10}])
            items_edited = st.data_editor(items_df, num_rows="dynamic", use_container_width=True, key="inv_items_edit", hide_index=True)
            st.markdown("**Payment / Bank**")
            ipc1, ipc2 = st.columns(2)
            with ipc1:
                pay_method = st.selectbox("支払方法 / Payment method",
                                          ["bank_transfer", "credit_card", "cash", "konbini", "other"],
                                          index=["bank_transfer","credit_card","cash","konbini","other"].index(
                                              i_default.get("payment_method","bank_transfer")))
                bank = st.text_input("銀行名 / Bank", i_default.get("bank_name","みずほ銀行"))
                branch = st.text_input("支店 / Branch", i_default.get("bank_branch","大手町支店"))
            with ipc2:
                acct_type = st.selectbox("口座種別 / Account type", ["ordinary","savings","checking"],
                                         index=["ordinary","savings","checking"].index(
                                             i_default.get("bank_account_type","ordinary")))
                acct_no = st.text_input("口座番号 / Account #", i_default.get("bank_account_number","1234567"))
                acct_holder = st.text_input("口座名義 / Holder (カタカナ)", i_default.get("bank_account_holder","カ）カナメ"))
            notes = st.text_area("備考 / Notes", i_default.get("notes","お支払期限を過ぎた場合、遅延損害金を申し受けます。"), height=60)
            inv_fmt = st.selectbox("Format", ["docx", "pdf"], index=0, key="inv_fmt")
            gen_inv = st.form_submit_button("📄 Generate 請求書", type="primary", use_container_width=True)
        if gen_inv:
            items_clean = []
            for r in items_edited.to_dict(orient="records"):
                try:
                    items_clean.append({
                        "description": str(r.get("description","")),
                        "quantity": int(r.get("quantity", 1) or 1),
                        "unit_price": float(r.get("unit_price", 0) or 0),
                        "tax_rate": float(r.get("tax_rate", default_tax) or default_tax),
                    })
                except Exception as e:
                    st.warning(f"Skipped bad row {r}: {e}")
            payload = {
                "invoice_number": inv_no, "issue_date": inv_date.isoformat(),
                "due_date": due_date.isoformat(),
                "seller_name": seller, "seller_address": seller_addr,
                "seller_phone": seller_tel, "seller_t_number": seller_tax,
                "buyer_name": buyer, "buyer_address": buyer_addr,
                "buyer_department": buyer_dep,
                "title": title, "subtitle": subtitle,
                "items": items_clean, "default_tax_rate": float(default_tax),
                "payment_method": pay_method,
                "bank_name": bank, "bank_branch": branch,
                "bank_account_type": acct_type, "bank_account_number": acct_no,
                "bank_account_holder": acct_holder,
                "notes": notes, "doc_format": inv_fmt,
            }
            _doc_generate_and_download("invoice", payload, inv_fmt, auto_translate_docs)

    # ---------------- TAB: 一般報告書 (Business report) ----------------
    with tab_report:
        st.markdown("### 📘 一般報告書 (Business Report — Ippan Hokokusho)")
        b_default = OFFICIAL_TEMPLATE_SAMPLES["business-report"]
        with st.form("biz_report_form"):
            br1, br2 = st.columns(2)
            with br1:
                br_title = st.text_input("報告書名 / Title", b_default.get("title","2026年度 第1四半期 業務報告書"))
                br_subtitle = st.text_input("副題 / Subtitle", b_default.get("subtitle","AI事業部 開発チーム"))
                br_author = st.text_input("作成者 / Author", b_default.get("author","山田 太郎"))
                br_dept = st.text_input("作成部署 / Author dept", b_default.get("author_department","AI事業部 開発部"))
                br_date = st.date_input("報告日 / Report date",
                                        datetime.strptime(b_default.get("report_date","2026-03-31"),"%Y-%m-%d").date())
            with br2:
                br_recp = st.text_input("提出先 / Recipient", b_default.get("recipient","鈴木 事業部長"))
                br_recp_dept = st.text_input("提出先部署 / Recipient dept", b_default.get("recipient_department","AI事業部"))
            br_exec = st.text_area("1. Executive Summary / 概要",
                                   b_default.get("executive_summary","新規AIプロダクト2件をリリースし、売上目標120%を達成した。"), height=90)
            br_bg = st.text_area("2. Background / 背景",
                                 b_default.get("background","2025年度末より進めていたLLM基盤を本番リリースした。"), height=70)
            st.markdown("**3. Objectives / 目的 (list)**")
            obj_rows = [o for o in (b_default.get("objectives") or []) if o]
            obj_df = pd.DataFrame(obj_rows, columns=["objective"]) if obj_rows else pd.DataFrame([{"objective":"目標 1"}])
            obj_edit = st.data_editor(obj_df, num_rows="dynamic", key="br_obj_edit", hide_index=True, use_container_width=True)
            br_methods = st.text_area("4. Methods / 方法", b_default.get("methods","アジャイル開発 × 6スプリント。"), height=70)
            br_res = st.text_area("5. Results / 結果",
                                  b_default.get("results","売上高前年同期比+22%、新規顧客12社獲得。"), height=90)
            br_disc = st.text_area("6. Discussion / 考察",
                                   b_default.get("discussion","要因は①既存製品満足度向上 ②LLM基盤採用加速 ③販売連携。"), height=70)
            st.markdown("**7. Issues / 課題**")
            iss_rows = [o for o in (b_default.get("issues") or []) if o]
            iss_df = pd.DataFrame(iss_rows, columns=["issue"]) if iss_rows else pd.DataFrame([{"issue":"課題 1"}])
            iss_edit = st.data_editor(iss_df, num_rows="dynamic", key="br_iss_edit", hide_index=True, use_container_width=True)
            br_concl = st.text_area("8. Conclusion / 結論",
                                    b_default.get("conclusion","通期計画に対し前倒しで進捗している。"), height=70)
            st.markdown("**9. Recommendations / 提言**")
            rec_rows = [o for o in (b_default.get("recommendations") or []) if o]
            rec_df = pd.DataFrame(rec_rows, columns=["recommendation"]) if rec_rows else pd.DataFrame([{"recommendation":"提言 1"}])
            rec_edit = st.data_editor(rec_df, num_rows="dynamic", key="br_rec_edit", hide_index=True, use_container_width=True)
            st.markdown("**10. References / 参考文献 (optional)**")
            ref_rows = [o for o in (b_default.get("references") or []) if o]
            ref_df = pd.DataFrame(ref_rows, columns=["reference"]) if ref_rows else pd.DataFrame(columns=["reference"])
            ref_edit = st.data_editor(ref_df, num_rows="dynamic", key="br_ref_edit", hide_index=True, use_container_width=True)
            st.markdown("**11. Attachments / 別紙 (optional)**")
            att_rows = [o for o in (b_default.get("attachments") or []) if o]
            att_df = pd.DataFrame(att_rows, columns=["attachment"]) if att_rows else pd.DataFrame(columns=["attachment"])
            att_edit = st.data_editor(att_df, num_rows="dynamic", key="br_att_edit", hide_index=True, use_container_width=True)
            br_fmt = st.selectbox("Format", ["docx", "pdf"], index=0, key="br_fmt")
            gen_br = st.form_submit_button("📄 Generate 一般報告書", type="primary", use_container_width=True)
        if gen_br:
            def _first(df, col):
                vals = df[col].dropna().astype(str).tolist() if col in df.columns else []
                return [v.strip() for v in vals if v and v.strip()]

            payload = {
                "title": br_title, "subtitle": br_subtitle, "author": br_author,
                "author_department": br_dept, "report_date": br_date.isoformat(),
                "recipient": br_recp, "recipient_department": br_recp_dept,
                "executive_summary": br_exec, "background": br_bg,
                "objectives": _first(obj_edit, "objective"), "methods": br_methods,
                "results": br_res, "discussion": br_disc,
                "issues": _first(iss_edit, "issue"), "conclusion": br_concl,
                "recommendations": _first(rec_edit, "recommendation"),
                "references": _first(ref_edit, "reference"),
                "attachments": _first(att_edit, "attachment"),
                "doc_format": br_fmt,
            }
            _doc_generate_and_download("business-report", payload, br_fmt, auto_translate_docs)

    # ---------------- TAB: 送付状 (Cover letter) ----------------
    with tab_cover:
        st.markdown("### ✉ 送付状 (Cover Letter / Transmittal — Soufu-Jo)")
        cl_default = OFFICIAL_TEMPLATE_SAMPLES["cover-letter"]
        with st.form("cover_form"):
            cc1, cc2 = st.columns(2)
            with cc1:
                s_company = st.text_input("差出人 会社名 / Sender company",
                                          cl_default.get("sender_company","KANAME株式会社"))
                s_dept = st.text_input("差出人 部署 / Sender dept", cl_default.get("sender_department","開発部"))
                s_name = st.text_input("差出人 氏名 / Sender name", cl_default.get("sender_name","山田 太郎"))
                s_zip = st.text_input("差出人 郵便番号 / Sender postal code",
                                      cl_default.get("sender_postal_code","100-0001"))
                s_addr = st.text_area("差出人 住所 / Sender address",
                                      cl_default.get("sender_address","東京都千代田区大手町1-1-1 KANAMEビル"), height=60)
                s_tel = st.text_input("差出人 TEL", cl_default.get("sender_phone","03-1234-5678"))
                s_email = st.text_input("差出人 Email", cl_default.get("sender_email","taro.yamada@kaname.co.jp"))
            with cc2:
                r_company = st.text_input("宛名 会社名 / Recipient company",
                                          cl_default.get("recipient_company","テスト商事株式会社"))
                r_dept = st.text_input("宛名 部署 / Recipient dept",
                                       cl_default.get("recipient_department","人事部 採用担当"))
                r_title = st.text_input("宛名 役職 / Title", cl_default.get("recipient_title","ご担当者様"))
                r_name = st.text_input("宛名 御中 / Name", cl_default.get("recipient_name","御中"))
                l_date = st.date_input("作成日 / Letter date",
                                       datetime.strptime(cl_default.get("letter_date","2026-03-31"), "%Y-%m-%d").date())
                subject = st.text_input("件名 / Subject", cl_default.get("subject","応募書類送付のご案内"))
            greeting = st.text_input("頭語 / Greeting", cl_default.get("greeting","お世話になっております。"))
            st.markdown("**本文 / Body paragraphs**")
            bp_rows = [p for p in (cl_default.get("body_paragraphs") or []) if p]
            bp_df = pd.DataFrame(bp_rows, columns=["paragraph"]) if bp_rows else pd.DataFrame([{"paragraph":"本文 1 段落"}])
            bp_edit = st.data_editor(bp_df, num_rows="dynamic", key="bp_edit", hide_index=True, use_container_width=True)
            closing = st.text_input("結語 / Closing", cl_default.get("closing","今後とも何卒よろしくお願い申し上げます。"))
            enc_title = st.text_input("添付表題 / Enclosures title",
                                      cl_default.get("enclosures_title","添付書類一覧"))
            st.markdown("**添付書類 / Enclosures list**")
            enc_rows = [p for p in (cl_default.get("enclosures") or []) if p]
            enc_df = pd.DataFrame(enc_rows, columns=["enclosure"]) if enc_rows else pd.DataFrame([{"enclosure":"履歴書 ×1"}])
            enc_edit = st.data_editor(enc_df, num_rows="dynamic", key="enc_edit", hide_index=True, use_container_width=True)
            cl_fmt = st.selectbox("Format", ["docx", "pdf"], index=0, key="cl_fmt")
            gen_cl = st.form_submit_button("📄 Generate 送付状", type="primary", use_container_width=True)
        if gen_cl:
            def _first(df, col):
                vals = df[col].dropna().astype(str).tolist() if col in df.columns else []
                return [v.strip() for v in vals if v and v.strip()]
            payload = {
                "sender_company": s_company, "sender_department": s_dept,
                "sender_name": s_name, "sender_postal_code": s_zip,
                "sender_address": s_addr, "sender_phone": s_tel, "sender_email": s_email,
                "recipient_company": r_company, "recipient_department": r_dept,
                "recipient_title": r_title, "recipient_name": r_name,
                "letter_date": l_date.isoformat(),
                "subject": subject, "greeting": greeting,
                "body_paragraphs": _first(bp_edit, "paragraph"),
                "closing": closing, "enclosures_title": enc_title,
                "enclosures": _first(enc_edit, "enclosure"),
                "doc_format": cl_fmt,
            }
            _doc_generate_and_download("cover-letter", payload, cl_fmt, auto_translate_docs)

    # ---------------- TAB: Expert JSON (original preserved, advanced users) ----------------
    with tab_expert:
        st.markdown("### 🧪 Expert — Raw JSON Payload Mode (Advanced)")
        st.caption("Select template → Load sample JSON → Edit → Generate + Download. Skips per-template form validations.")

        csel, cfmt, cbtn = st.columns([3, 1, 1])
        with csel:
            tmpl_id = st.selectbox("Template", list(TEMPLATE_LABELS.keys()),
                                   format_func=lambda x: TEMPLATE_LABELS[x][0], key="exp_tmpl")
        with cfmt:
            tpl_fmt = st.selectbox("Format", ["docx", "pdf"], index=0, key="exp_tpl_fmt")
        with cbtn:
            st.markdown("<div style='height:26px'></div>", unsafe_allow_html=True)
            load_sample = st.button("⟳ Load sample data", use_container_width=True, key="exp_load")

        lbl, desc = TEMPLATE_LABELS[tmpl_id]
        st.info(f"**{lbl}** — {desc}")

        default_json = json.dumps(OFFICIAL_TEMPLATE_SAMPLES[tmpl_id], ensure_ascii=False, indent=2)
        tpl_payload_raw = st.text_area(
            "Template JSON payload (edit fields as needed)",
            value=default_json if not load_sample else json.dumps(OFFICIAL_TEMPLATE_SAMPLES[tmpl_id], ensure_ascii=False, indent=2),
            height=400, key=f"exp_tpl_json_{tmpl_id}",
            help="Valid JSON matching the backend Pydantic schema for this template.",
        )

        col_meta, col_dl = st.columns(2)
        with col_meta:
            run_meta = st.button("🔎 Generate (metadata only)", use_container_width=True, key="exp_meta_btn")
        with col_dl:
            run_dl = st.button("📥 Generate + Download file", type="primary", use_container_width=True, key="exp_dl_btn")

        def _try_parse_json(raw: str):
            try:
                return True, json.loads(raw)
            except Exception as e:
                return False, f"Invalid JSON: {e}"

        if (run_meta or run_dl) and tpl_payload_raw.strip():
            ok_parse, obj = _try_parse_json(tpl_payload_raw)
            if not ok_parse:
                st.error(obj)
            elif not _REQUESTS_AVAILABLE:
                st.error("requests not installed")
            else:
                path = (
                    f"/api/documents/generate/{tmpl_id}/download"
                    if run_dl
                    else f"/api/documents/generate/{tmpl_id}"
                )
                with st.spinner(f"Generating {tmpl_id} · {tpl_fmt.upper()} via backend..."):
                    ok, resp = _request(
                        "POST", path,
                        payload=obj,
                        params={"auto_translate": "true" if auto_translate_docs else "false"},
                        timeout=120, raw=run_dl,
                    )
                if ok and run_dl:
                    r = resp
                    content_type = r.headers.get("Content-Type", "")
                    disp = r.headers.get("Content-Disposition", "")
                    fname = _extract_filename_from_disp(disp, f"{tmpl_id}_output.{tpl_fmt}")
                    bytes_data = r.content
                    st.success(f"✅ Generated · {len(bytes_data):,} bytes.")
                    meta_row = st.columns(4)
                    meta_row[0].caption(f"Template: `{tmpl_id}`")
                    meta_row[1].caption(f"Format: {tpl_fmt.upper()}")
                    meta_row[2].caption(f"Type: `{content_type}`")
                    meta_row[3].caption(f"Size: {len(bytes_data):,} B")
                    st.download_button(
                        label=f"⬇ Download {fname}",
                        data=bytes_data,
                        file_name=fname,
                        mime=content_type or "application/octet-stream",
                        type="primary",
                        use_container_width=True,
                    )
                elif ok:
                    st.json(resp)
                else:
                    st.error(f"Generation failed: {resp}")

    # ---------------- TAB: Recent Documents (ledger view) ----------------
    with tab_recent:
        st.markdown("### 📂 Recent Documents (SQLite Ledger)")
        if not _REQUESTS_AVAILABLE:
            st.warning("Backend unreachable")
        else:
            limit_n = st.slider("Recent count", 10, 200, 50, key="recent_limit")
            with st.spinner("Fetching recent documents..."):
                ok_d, dlist = _request("GET", f"/api/documents?limit={limit_n}")
            if not ok_d:
                st.error(str(dlist))
            else:
                docs = dlist.get("documents") if isinstance(dlist, dict) else []
                if not docs:
                    st.caption("No documents recorded yet. DB ledger may be offline.")
                else:
                    st.success(f"Loaded {len(docs)} recent documents.")
                    rows = []
                    for d in docs:
                        rows.append({
                            "id": (d.get("document_id") or "")[:10],
                            "generator": d.get("generator"),
                            "title": d.get("title"),
                            "format": d.get("format"),
                            "size_bytes": d.get("size_bytes"),
                            "author": d.get("author"),
                            "created_at": d.get("created_at"),
                        })
                    rdf = pd.DataFrame(rows)
                    st.data_editor(rdf, use_container_width=True, hide_index=True, key="recent_docs_edit")
                    with st.expander("Full JSON of most recent 5 documents", expanded=False):
                        for d in docs[:5]:
                            st.json(d)

    with tab_tools:
        st.markdown("### AI-Assisted Outline")
        st.caption("Uses the configured LLM provider. This endpoint does not silently substitute a non-AI outline.")
        outline_topic = st.text_input("Document topic", key="outline_topic")
        outline_audience = st.text_input("Audience", key="outline_audience")
        outline_language = st.selectbox("Outline language", ["ja", "en"], key="outline_language")
        if st.button("Draft Outline", key="outline_draft") and _REQUESTS_AVAILABLE:
            ok_outline, outline = _request(
                "POST", "/api/documents/outline/draft",
                payload={
                    "topic": outline_topic, "audience": outline_audience,
                    "language": outline_language, "section_count": 5,
                },
                timeout=180,
            )
            if ok_outline and isinstance(outline, dict):
                st.caption(f"Provider: {outline.get('provider')}")
                st.text_area("Draft outline", outline.get("outline", ""), height=220, key="outline_result")
            else:
                st.warning(str(outline))

        st.markdown("### Template Batch Generation")
        st.caption("Provide 1-20 validated template/payload items. Output is an all-or-nothing ZIP with a manifest.")
        batch_default = json.dumps(
            [{"template_id": "custom-document", "payload": {
                "title": "Batch Document", "sections": [{"heading": "Overview", "body": "Sample content"}],
                "doc_format": "docx",
            }}],
            ensure_ascii=False,
            indent=2,
        )
        batch_json = st.text_area("Batch items JSON", value=batch_default, height=220, key="document_batch_json")
        batch_auto_translate = st.checkbox("Auto-translate English fields", value=False, key="document_batch_translate")
        if st.button("Generate Batch ZIP", key="document_batch_run") and _REQUESTS_AVAILABLE:
            try:
                items = json.loads(batch_json)
                if not isinstance(items, list):
                    raise ValueError("Input must be a JSON array.")
                ok_batch, batch_response = _request(
                    "POST", "/api/documents/batch/generate/download",
                    payload={"items": items, "auto_translate": batch_auto_translate},
                    timeout=300,
                    raw=True,
                )
                if ok_batch and hasattr(batch_response, "content"):
                    st.download_button(
                        "Download generated documents",
                        data=batch_response.content,
                        file_name="kaname_documents.zip",
                        mime="application/zip",
                        key="document_batch_download",
                    )
                else:
                    st.error(batch_response.text if hasattr(batch_response, "text") else str(batch_response))
            except (ValueError, json.JSONDecodeError) as exc:
                st.error(f"Invalid batch JSON: {exc}")

        st.markdown("### Translate DOCX")
        docx_file = st.file_uploader("DOCX input", type=["docx"], key="docx_translate_file")
        docx_source = st.selectbox("DOCX source language", ["auto", "ja", "en"], key="docx_translate_source")
        docx_target = st.selectbox("DOCX target language", ["ja", "en"], key="docx_translate_target")
        if st.button("Translate DOCX", key="docx_translate_run") and docx_file and _REQUESTS_AVAILABLE:
            ok_docx, docx_response = _request(
                "POST", "/api/documents/translate-docx",
                files={"file": (docx_file.name, docx_file.getvalue(), docx_file.type)},
                data={"source_lang": docx_source, "target_lang": docx_target},
                timeout=300,
                raw=True,
            )
            if ok_docx and hasattr(docx_response, "content"):
                st.download_button(
                    "Download translated DOCX",
                    data=docx_response.content,
                    file_name=f"{os.path.splitext(docx_file.name)[0]}_{docx_target}.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    key="docx_translate_download",
                )
            else:
                st.error(docx_response.text if hasattr(docx_response, "text") else str(docx_response))

        st.markdown("### Merge Japanese / English PDFs")
        pdf_files = st.file_uploader(
            "Choose 2-20 PDFs. Pages are merged in the selected upload order.",
            type=["pdf"],
            accept_multiple_files=True,
            key="pdf_merge_files",
        )
        if st.button("Merge PDFs", key="pdf_merge_run") and pdf_files and _REQUESTS_AVAILABLE:
            try:
                files = [("files", (item.name, item.getvalue(), "application/pdf")) for item in pdf_files]
                response = requests.post(
                    f"{BACKEND_URL}/api/documents/merge-pdfs",
                    files=files,
                    timeout=180,
                )
                if response.ok:
                    st.download_button(
                        "Download merged PDF",
                        data=response.content,
                        file_name="kaname_bilingual_merged.pdf",
                        mime="application/pdf",
                        key="pdf_merge_download",
                    )
                else:
                    st.error(f"PDF merge failed: HTTP {response.status_code}: {response.text[:300]}")
            except Exception as exc:
                st.error(f"PDF merge failed: {exc}")


# ---------------------------------------------------------------------------
# MODULE: MACHINE LEARNING
# ---------------------------------------------------------------------------

ML_CLASSIFY_SAMPLES = [
    ("技術 - AI/ML", "深層学習モデルをGPUクラスタ上で分散学習させ、推論レイテンシを12msに改善しました。"),
    ("ビジネス・文書", "本件につきましては、来週中に契約書を締結させていただく所存です。"),
    ("学術・研究", "被験者24名に対してfMRI計測を行い、線形判別分析により分類精度78%を達成した。"),
    ("日常・ソーシャル", "今日のランチはイタリアンにしようと思う。美味しいパスタが食べたい。"),
    ("法律・規制", "本法第12条に基づき、当事者は遅滞なく書面による通知を行うものとする。"),
]

ML_CLUSTER_SAMPLES = """\
Pythonでscikit-learnを使って分類モデルを構築しました。
深層学習モデルをGPUで学習させ、推論APIとして公開しました。
Kubernetesクラスタにマイクロサービスをデプロイし、HPAを設定しました。
来週の契約締結に向けて、特許使用許諾書を確認中です。
本店監査役は、第12条に基づき報告書を提出しなければならない。
今回の事件は民法上の不法行為に該当する可能性があります。
被験者18名に対しアンケート調査を行い、因子分析を実施しました。
この研究結果は、統計学的に有意水準5%で効果が確認されました。
昨日の映画はとても面白かった。次の週末も観に行きたい。
近くのイタリアンレストランのパスタがめちゃくちゃ美味しかった。"""

ML_INTENT_SAMPLES = [
    ("ご挨拶", "お世話になっております。KANAME株式会社の山田でございます。"),
    ("質問", "先日送付いただいた見積書について、いくつか確認したい点があります。"),
    ("依頼", "来週金曜日までに、改訂版の仕様書をご提出いただけますでしょうか。"),
    ("命令・指示", "このディレクトリをzipで固めて、添付ファイルで送付してください。"),
    ("意見・感想", "今回の提案は非常に魅力的だと思います。特にコスト削減効果が大きい点が優れています。"),
    ("その他", "2026年3月31日現在の売上高は前年比122%となりました。"),
]


def render_ml():
    st.markdown("## Machine Learning — Japanese Text Intelligence")
    st.markdown('<span class="status-badge status-impl">Implemented — Sudachi TF-IDF + scikit-learn Pipeline (Classification / Clustering / Intent)</span>',
                unsafe_allow_html=True)
    st.markdown("")

    if _REQUESTS_AVAILABLE:
        ok, msg = _request("GET", "/api/ml/health")
        if ok and isinstance(msg, dict):
            st.success(
                f"Engines: {', '.join(msg.get('engines') or ['?'])}  ·  "
                f"Classes × {len(msg.get('classes') or [])}" if msg.get('classes') else
                f"Engines: {', '.join(msg.get('engines') or ['?'])}"
            )
        else:
            st.error(msg if isinstance(msg, str) else "Backend offline")

    with st.expander("About this module", expanded=False):
        st.markdown("""
            **Training data:** 5 Japanese text categories × ~5 samples = 25 short texts (for demo only).
            **Classifier:** CalibratedClassifierCV(LinearSVC) + TF-IDF(Sudachi tokenizer, POS filter, length filter, JA stopwords)
            **Clustering:** KMeans over TF-IDF, with top keywords per cluster and representative example sentences
            **Intent Detection (6 classes):** Rule-early hybrid → regex pattern matchers (greeting/question/request/command/opinion) + calibrated ML classifier fallback with method=rule or method=classifier indicator
        """)

    tab_cls, tab_clu, tab_int, tab_torch, tab_advanced = st.tabs([
        "🏷 Classification", "🧩 Clustering", "💬 Intent Detection",
        "🧠 Custom Model (PyTorch)", "🧪 Topics + Auto-label",
    ])

    with tab_cls:
        c_sample, c_topk = st.columns([3, 1])
        with c_sample:
            cls_sel = st.selectbox("Sample text", [s[0] for s in ML_CLASSIFY_SAMPLES], index=0,
                                   format_func=lambda x: x, key="cls_samp")
        with c_topk:
            top_k = st.slider("Top-K labels", min_value=2, max_value=5, value=5, step=1, key="cls_topk")
        sample_text = next(s[1] for s in ML_CLASSIFY_SAMPLES if s[0] == cls_sel)
        cls_txt = st.text_area("Text to classify", value=sample_text, height=130, key="cls_txt")
        run_cls = st.button("🏷 Classify Text", type="primary", use_container_width=True, key="cls_run")

        with st.expander("ℹ Model info", expanded=False):
            if _REQUESTS_AVAILABLE:
                ok_info, info = _request("GET", "/api/ml/classifier/info")
                if ok_info and isinstance(info, dict):
                    ic1, ic2, ic3, ic4 = st.columns(4)
                    ic1.metric("Classes", info.get("num_classes", "?"))
                    ic2.metric("Features", info.get("num_features", "?"))
                    ic3.metric("Samples", info.get("training_samples", "?"))
                    ic4.metric("Calibration", "Platt" if info.get("calibrated") else "None")
                    st.write("**Classes:**")
                    st.markdown("  ·  ".join(f"`{c}`" for c in info.get("classes", [])))
                    st.write("**Method:** CalibratedClassifierCV(LinearSVC, sigmoid) + Sudachi TF-IDF Vectorizer")

        if run_cls and cls_txt.strip() and _REQUESTS_AVAILABLE:
            with st.spinner("Classifying text..."):
                ok_c, r_c = _request("POST", "/api/ml/classify",
                                     payload={"text": cls_txt, "top_k": top_k})
            if ok_c and isinstance(r_c, dict):
                col_lab, col_conf = st.columns([2, 3])
                with col_lab:
                    st.markdown(f"### Predicted: `{r_c.get('predicted_label', '?')}`")
                    st.metric("Confidence", f"{float(r_c.get('confidence',0))*100:.1f}%")
                probs = r_c.get("probabilities") or {}
                if probs:
                    df_c = pd.DataFrame({
                        "label": list(probs.keys()),
                        "probability": [float(v) for v in probs.values()],
                    }).sort_values("probability", ascending=True)
                    with col_conf:
                        fig = px.bar(df_c, x="probability", y="label", orientation="h",
                                     title="Top-K label probability",
                                     color="probability", color_continuous_scale=_KANAME_SCALE,
                                     range_x=[0, 1.05])
                        fig.update_layout(height=260, margin=dict(l=0, r=10, t=40, b=0))
                        st.plotly_chart(fig, use_container_width=True)
                with st.expander("Raw JSON response"):
                    st.json(r_c)
            elif not ok_c:
                st.error(f"Classification failed: {r_c}")

    with tab_clu:
        st.caption("Paste newline-separated Japanese sentences; the module will cluster them with KMeans and return top keywords + representative examples per cluster.")
        n_clu = st.slider("N clusters", min_value=2, max_value=6, value=3, step=1, key="clu_n")
        clu_txt = st.text_area("Newline-separated sentences", value=ML_CLUSTER_SAMPLES, height=220, key="clu_txt")
        rdemo, rrun = st.columns(2)
        run_demo = rdemo.button("🎲 Run on built-in demo corpus", use_container_width=True, key="clu_demo")
        run_clu = rrun.button("🧩 Cluster my texts", type="primary", use_container_width=True, key="clu_run")

        def _render_clusters(resp):
            if not isinstance(resp, dict):
                return
            clusters = resp.get("clusters") or []
            st.success(f"Clustered {resp.get('total_texts', '?')} texts into {resp.get('n_clusters', '?')} clusters.")
            for cl in clusters:
                idx = cl.get("cluster_id", 0)
                size = cl.get("size", 0)
                topk = cl.get("top_keywords") or []
                examples = cl.get("example_sentences") or []
                with st.expander(f"🟣 Cluster {idx+1}  ·  size={size}  ·  top: {', '.join(topk[:6])}", expanded=True):
                    kws_s = pd.Series({k: 1/(i+1) for i, k in enumerate(topk[:10])})
                    if _PLOTLY_AVAILABLE and len(kws_s) > 0:
                        fig = px.bar(x=kws_s.index, y=kws_s.values,
                                     title=f"Cluster {idx+1} top keywords",
                                     labels={"x": "keyword", "y": "weight (1/rank)"},
                                     color=kws_s.values, color_continuous_scale=_KANAME_SCALE)
                        fig.update_layout(height=220, margin=dict(l=0, r=10, t=40, b=0))
                        st.plotly_chart(fig, use_container_width=True)
                    st.write("**Example sentences:**")
                    for s in examples:
                        st.write(f"- {s}")

        if run_clu and clu_txt.strip() and _REQUESTS_AVAILABLE:
            texts = [line.strip() for line in clu_txt.splitlines() if line.strip()]
            with st.spinner(f"Clustering {len(texts)} texts into {n_clu} clusters..."):
                ok, r = _request("POST", "/api/ml/cluster",
                                 payload={"texts": texts, "n_clusters": n_clu, "top_keywords": 10, "n_examples": 5})
            if ok:
                _render_clusters(r)
            else:
                st.error(f"Clustering failed: {r}")
        if run_demo and _REQUESTS_AVAILABLE:
            with st.spinner("Running built-in cluster demo..."):
                ok, r = _request("GET", f"/api/ml/cluster/demo?n_clusters={n_clu}")
            if ok:
                _render_clusters(r)
            else:
                st.error(f"Demo failed: {r}")

    with tab_int:
        int_sel = st.selectbox("Sample", [s[0] for s in ML_INTENT_SAMPLES], index=0, key="int_sel")
        int_sample = next(s[1] for s in ML_INTENT_SAMPLES if s[0] == int_sel)
        int_txt = st.text_area("Text to detect intent", value=int_sample, height=120, key="int_txt")
        run_int = st.button("💬 Detect Intent", type="primary", use_container_width=True, key="int_run")

        with st.expander("ℹ Intent classifier info", expanded=False):
            if _REQUESTS_AVAILABLE:
                ok, info = _request("GET", "/api/ml/intent/info")
                if ok and isinstance(info, dict):
                    rule_counts = info.get("rule_patterns") or {}
                    ic1, ic2, ic3 = st.columns(3)
                    ic1.metric("Classes", len(info.get("classes", [])))
                    ic2.metric("Rule patterns", sum(rule_counts.values()) if rule_counts else "?")
                    ic3.metric("Train pairs", info.get("training_samples", "?"))
                    st.write("**Classes:** " + "  ·  ".join(f"`{c}`" for c in info.get("classes", [])))
                    st.caption("Pipeline: Rule pattern matcher (fast) → calibrated classifier fallback when no rule hits.")

        with st.expander("📊 Accuracy evaluation", expanded=False):
            if _REQUESTS_AVAILABLE and st.button("Run evaluation", key="int_eval"):
                with st.spinner("Cross-validating intent model..."):
                    ok_e, ev = _request("GET", "/api/ml/intent/evaluate", timeout=180)
                if ok_e and isinstance(ev, dict):
                    e1, e2, e3 = st.columns(3)
                    e1.metric("Hybrid accuracy", f"{float(ev.get('hybrid_accuracy', 0))*100:.1f}%")
                    e2.metric("ML-only (CV) accuracy", f"{float(ev.get('accuracy', 0))*100:.1f}%")
                    e3.metric("Samples", ev.get("samples", "?"))
                    per = ev.get("hybrid_per_class_accuracy") or {}
                    if per:
                        st.markdown("**Per-class accuracy (hybrid)**")
                        st.dataframe(
                            pd.DataFrame({"intent": list(per.keys()),
                                          "accuracy": [float(v) for v in per.values()]}),
                            use_container_width=True, hide_index=True,
                        )
                else:
                    st.error(ev if isinstance(ev, str) else "Evaluation unavailable")

        if run_int and int_txt.strip() and _REQUESTS_AVAILABLE:
            with st.spinner("Detecting intent..."):
                ok, r = _request("POST", "/api/ml/intent", payload={"text": int_txt})
            if ok and isinstance(r, dict):
                int_m1, int_m2, int_m3 = st.columns(3)
                int_m1.metric("Intent", r.get("intent", "?"))
                int_m2.metric("Confidence", f"{float(r.get('confidence', 0))*100:.1f}%")
                method = r.get("method", "?")
                badge_color = "#16a34a" if method == "rule" else ("#6d28d9" if method == "classifier" else "#78716c")
                int_m3.markdown(
                    f"<span class='status-badge' style='background:{badge_color};color:white'>Method: {method}"
                    + (f"  ·  rule_hit=`{r.get('rule_hit','')}`" if r.get("rule_hit") else "")
                    + "</span>", unsafe_allow_html=True,
                )
                probs = r.get("probabilities") or {}
                if probs and _PLOTLY_AVAILABLE:
                    df_i = pd.DataFrame({
                        "intent": list(probs.keys()),
                        "probability": [float(v) for v in probs.values()],
                    }).sort_values("probability", ascending=False)
                    fig = px.bar(df_i, x="intent", y="probability", title="Intent probability distribution",
                                 color="probability", color_continuous_scale=_KANAME_SCALE,
                                 range_y=[0, 1.05])
                    fig.update_layout(height=260, margin=dict(l=0, r=10, t=40, b=0))
                    st.plotly_chart(fig, use_container_width=True)
                with st.expander("Raw JSON"):
                    st.json(r)
            elif not ok:
                st.error(f"Intent detection failed: {r}")

    with tab_torch:
        st.caption("A custom neural network (MLP) trained end-to-end with PyTorch on the built-in Japanese corpus — not an API wrapper. Weights are learned by gradient descent and cached to disk.")
        torch_info = None
        if _REQUESTS_AVAILABLE:
            ok_ti, ti = _request("GET", "/api/ml/torch/info", timeout=120)
            if ok_ti and isinstance(ti, dict) and ti.get("available"):
                torch_info = ti
            elif ok_ti and isinstance(ti, dict):
                st.warning(f"Custom PyTorch model unavailable: {ti.get('error', 'unknown error')}")
            else:
                st.error(ti if isinstance(ti, str) else "Backend offline")

        if torch_info:
            with st.expander("ℹ Custom model info", expanded=True):
                tc1, tc2, tc3, tc4 = st.columns(4)
                tc1.metric("Engine", torch_info.get("engine", "?"))
                tc2.metric("Torch", torch_info.get("torch_version", "?"))
                tc3.metric("Features", torch_info.get("num_features", "?"))
                acc = torch_info.get("train_accuracy")
                tc4.metric("Train acc", f"{float(acc)*100:.1f}%" if acc is not None else "?")
                st.write("**Architecture:** " + str(torch_info.get("architecture", "?")))
                st.write("**Optimizer:** " + str(torch_info.get("optimizer", "?")) + "  ·  **Vectorizer:** " + str(torch_info.get("vectorizer", "?")))
                st.write("**Classes:** " + "  ·  ".join(f"`{c}`" for c in torch_info.get("classes", [])))
                if st.button("🔁 Retrain model", key="torch_retrain"):
                    with st.spinner("Training MLP via gradient descent..."):
                        ok_tr, tr = _request("POST", "/api/ml/torch/train", payload={}, timeout=300)
                    if ok_tr and isinstance(tr, dict):
                        st.success(f"Retrained — train accuracy {float(tr.get('train_accuracy', 0))*100:.1f}% (loss {tr.get('final_loss')})")
                        st.rerun()
                    else:
                        st.error(tr if isinstance(tr, str) else "Training failed")

            t_sel = st.selectbox("Sample text", [s[0] for s in ML_CLASSIFY_SAMPLES], index=0, key="torch_samp")
            t_sample = next(s[1] for s in ML_CLASSIFY_SAMPLES if s[0] == t_sel)
            t_txt = st.text_area("Text to classify (custom model)", value=t_sample, height=130, key="torch_txt")
            t_topk = st.slider("Top-K labels", min_value=2, max_value=5, value=3, step=1, key="torch_topk")
            run_torch = st.button("🧠 Classify with Custom Model", type="primary", use_container_width=True, key="torch_run")

            if run_torch and t_txt.strip():
                with st.spinner("Running neural classifier..."):
                    ok_tr, tr = _request("POST", "/api/ml/torch/classify",
                                         payload={"text": t_txt, "top_k": t_topk}, timeout=120)
                if ok_tr and isinstance(tr, dict):
                    tcl, tcr = st.columns([2, 3])
                    with tcl:
                        st.markdown(f"### Predicted: `{tr.get('predicted_label', '?')}`")
                        st.metric("Confidence", f"{float(tr.get('confidence', 0))*100:.1f}%")
                    probs = tr.get("probabilities") or {}
                    if probs and _PLOTLY_AVAILABLE:
                        df_t = pd.DataFrame({
                            "label": list(probs.keys()),
                            "probability": [float(v) for v in probs.values()],
                        }).sort_values("probability", ascending=True)
                        with tcr:
                            fig_t = px.bar(df_t, x="probability", y="label", orientation="h",
                                           title="Custom model label probability",
                                           color="probability", color_continuous_scale=_KANAME_SCALE,
                                           range_x=[0, 1.05])
                            fig_t.update_layout(height=260, margin=dict(l=0, r=10, t=40, b=0))
                            st.plotly_chart(fig_t, use_container_width=True)
                    with st.expander("Raw JSON response"):
                        st.json(tr)
                elif not ok_tr:
                    st.error(f"Custom model classification failed: {tr}")

    with tab_advanced:
        st.markdown("### Japanese Topic Discovery")
        st.caption("Unsupervised LDA topics are exploratory; review topic keywords and assignments before using them as labels.")
        topic_text = st.text_area(
            "One document per line",
            value=ML_CLUSTER_SAMPLES,
            height=180,
            key="topic_texts",
        )
        topic_count = st.slider("Number of topics", 2, 10, 3, key="topic_count")
        if st.button("Discover Topics", key="topic_run") and _REQUESTS_AVAILABLE:
            docs = [line.strip() for line in topic_text.splitlines() if line.strip()]
            ok_topic, topic_result = _request(
                "POST", "/api/ml/topics",
                payload={"texts": docs, "n_topics": topic_count, "top_words": 8},
                timeout=120,
            )
            if ok_topic and isinstance(topic_result, dict):
                for item in topic_result.get("topics", []):
                    terms = item.get("keywords", [])
                    st.markdown(
                        f"**Topic {item.get('topic_id', 0) + 1}:** "
                        + " · ".join(term.get("term", "") for term in terms)
                    )
                st.dataframe(
                    pd.DataFrame(topic_result.get("assignments") or []),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.error(str(topic_result))

        st.markdown("### Confidence-Gated Auto-labeling")
        auto_text = st.text_area(
            "Unlabeled Japanese texts, one per line",
            value="機械学習モデルを構築します。\n今日は天気が良いです。",
            height=100,
            key="auto_label_texts",
        )
        auto_threshold = st.slider("Minimum confidence", 0.0, 1.0, 0.8, 0.05, key="auto_label_threshold")
        if st.button("Suggest Labels", key="auto_label_run") and _REQUESTS_AVAILABLE:
            docs = [line.strip() for line in auto_text.splitlines() if line.strip()]
            ok_labels, labels = _request(
                "POST", "/api/ml/auto-label",
                payload={"texts": docs, "confidence_threshold": auto_threshold},
                timeout=120,
            )
            if ok_labels and isinstance(labels, dict):
                st.warning(labels.get("warning", "Review suggestions before accepting."))
                st.dataframe(pd.DataFrame(labels.get("items") or []), use_container_width=True, hide_index=True)
            else:
                st.error(str(labels))

        st.markdown("### Custom PyTorch Transformer")
        st.caption("CPU demonstration model trained on the built-in 25-example corpus. Training accuracy is not held-out accuracy.")
        transformer_epochs = st.slider("Training epochs", 1, 300, 50, key="transformer_epochs")
        if st.button("Train Transformer", key="transformer_train") and _REQUESTS_AVAILABLE:
            with st.spinner("Training the TransformerEncoder on CPU..."):
                ok_transformer, transformer_info = _request(
                    "POST", "/api/ml/torch-transformer/train",
                    payload={"epochs": transformer_epochs, "seed": 42},
                    timeout=300,
                )
            if ok_transformer and isinstance(transformer_info, dict):
                st.session_state["kaname_transformer_trained"] = True
                st.success(
                    f"Trained on {transformer_info.get('training_samples')} examples · "
                    f"training accuracy {float(transformer_info.get('train_accuracy', 0)):.1%}"
                )
                st.json(transformer_info)
            else:
                st.error(str(transformer_info))

        transformer_text = st.text_input(
            "Text for Transformer classification",
            value=ML_CLASSIFY_SAMPLES[0][1],
            key="transformer_text",
        )
        if st.button("Classify with Transformer", key="transformer_classify") and _REQUESTS_AVAILABLE:
            with st.spinner("Running Transformer inference..."):
                ok_transformer, transformer_result = _request(
                    "POST", "/api/ml/torch-transformer/classify",
                    payload={"text": transformer_text, "top_k": 3},
                    timeout=180,
                )
            if ok_transformer and isinstance(transformer_result, dict):
                st.write(
                    f"**Prediction:** {transformer_result.get('predicted_label')} · "
                    f"confidence {float(transformer_result.get('confidence', 0)):.1%}"
                )
                st.caption("The endpoint will train the demonstration model on first use if no saved weights exist.")
            else:
                st.error(str(transformer_result))


# ---------------------------------------------------------------------------
# MODULE: PROJECT INFO
# ---------------------------------------------------------------------------

def render_project_info():
    st.markdown("## Project Information")
    st.markdown("")

    st.markdown("### KANAME [要]")
    st.write("**Full Name:** AI-Powered Japanese Language and Productivity Intelligence System")
    st.write("**Project Type:** MCA Minor Project — Semester-Long Individual AI Project")
    st.write("**Current Phase:** Phase 4 — Voice Intelligence · Custom ML Model · Durable Persistence · Supervised NER · Dataset Reuse (NLP · Voice · Data · Translation · Documents · ML · History)")
    st.caption("Translation: LLM hybrid (Claude / Gemini) + 4-domain glossary + request-scoped glossaries & few-shot examples + DOCX translation. "
               "ML: Sudachi TF-IDF + LinearSVC + KMeans + rule-early hybrid intent + LDA topics + confidence-gated auto-labeling + PyTorch Transformer + supervised BIO NER. "
               "Documents: NLP report + 5 official Japanese templates × DOCX/PDF + AI outline drafting + batch generation + bilingual PDF merge + company-analysis from datasets.")

    online, status, lat = backend_online()
    s1, s2, s3 = st.columns(3)
    s1.metric("Backend", "Online" if online else "Offline")
    s2.metric("Backend Status", status)
    s3.metric("Backend Latency", f"{int(lat * 1000)} ms" if online else "—")

    st.markdown("")
    st.markdown("### Technology Stack")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Core")
        st.markdown("""
            - **Language:** Python 3.10
            - **Main Frontend:** Streamlit (:8501)
            - **Voice / Audio UI:** Gradio (:7860)
            - **Backend:** FastAPI + Uvicorn (:8000)
            - **Database:** SQLite + SQLAlchemy 2.x + Alembic
            - **Virtual env:** `venv/` already installed
        """)
    with c2:
        st.markdown("#### AI & Data")
        st.markdown("""
            - **Japanese NLP:** SudachiPy (SudachiDict-core) + JA stopwords + sentiment lexicon
            - **Speech-to-Text:** faster-whisper (CTranslate2)
            - **Text-to-Speech:** pyttsx3 (optional, host TTS engine)
            - **Speaker Diarization:** pyannote.audio (optional, gated on HF_TOKEN)
            - **Translation:** Optional-LLM hybrid · Claude / Gemini auto-detect + dictionary fallback · 4 bilingual domain glossaries + request glossaries/few-shot examples
            - **ML:** scikit-learn · Sudachi TF-IDF · LinearSVC · CalibratedClassifierCV · KMeans · LDA · rule-early hybrid intent detection
            - **Deep Learning:** PyTorch (CPU) — TransformerEncoder demo classifier + supervised BIO NER
            - **Data:** pandas, NumPy, scikit-learn profiling (missingness, skew, IQR outliers, cardinality)
            - **Visualization:** Plotly
            - **Documents:** python-docx, ReportLab (HeiseiKakuGo-W5 JP CID font) · pypdf · RFC 5987 UTF-8 filenames
        """)

    st.markdown("")
    st.markdown("### Implemented Module Endpoints")
    with st.expander("Expand to see current API endpoints"):
        st.code("""
GET  /health                   → {"status":"KANAME backend is running"}

=== Japanese NLP ===
GET  /api/nlp/health           → SudachiPy engine status
POST /api/nlp/tokenize         → Sudachi tokenization
POST /api/nlp/analyze          → Full morphological analysis (w/ history)
POST /api/nlp/keywords         → TF-IDF keyword extraction (top_n, POS filtered)
POST /api/nlp/sentiment        → Rule-based sentiment analysis (JA lexicon)
POST /api/nlp/entities         → Sudachi dictionary/POS named-entity spans
GET  /api/nlp/entities/model   → Supervised NER model status
POST /api/nlp/entities/train   → Train supervised BIO NER from span annotations
POST /api/nlp/entities/predict → Predict entities with the supervised NER model
POST /api/nlp/analyze-jlpt     → JLPT-level vocabulary analysis
POST /api/nlp/alternatives     → 6-style Japanese sentence rewrites
GET  /api/nlp/history?limit=N  → Persisted NLP analyses from SQLite
GET  /api/nlp/analyses/{id}    → Single persisted analysis

=== Voice Intelligence ===
GET  /api/voice/health         → faster-whisper engine status
POST /api/voice/transcribe     → multipart audio upload → STT
POST /api/voice/transcribe-inline → JSON audio_path → STT
POST /api/voice/transcribe-then-nlp → STT + Sudachi combined
POST /api/voice/commands/parse → Voice command intent parsing
POST /api/voice/tts            → Text-to-Speech (pyttsx3, optional)
POST /api/voice/diarize        → Speaker diarization (pyannote, optional)
POST /api/voice/post-process   → Clean/format transcribed text
POST /api/voice/subtitles      → Generate subtitle (SRT/VTT) output
POST /api/voice/keyword-spot   → Keyword spotting in transcript
POST /api/voice/summarize      → Transcript summarization
GET  /api/voice/jobs?limit=N   → Persisted VoiceJob rows from SQLite
GET  /api/voice/jobs/{id}      → Single voice job

=== Data Intelligence ===
GET  /api/data/health          → analyzer status
POST /api/data/upload          → upload dataset (stores source bytes)
POST /api/data/analyze         → dataset → full summary + stats + samples
POST /api/data/profile         → data-quality profiling (missingness, skew, IQR outliers)
POST /api/data/correlation-matrix → correlation heatmap data
POST /api/data/outliers        → IQR outlier detection
POST /api/data/dashboard       → full EDA dashboard payload
GET  /api/data/datasets        → list stored dataset records
GET  /api/data/datasets/{id}   → retrieve stored dataset by dataset_id

=== Translation Intelligence ===
GET  /api/translation/health   → engine / domains / configured_keys
GET  /api/translation/engines  → LLM backend availability
GET  /api/translation/evaluate → translation quality evaluation
GET  /api/translation/glossaries         → 4 domain list
GET  /api/translation/glossaries/{dom}   → JA/EN pairs list
POST /api/translation/translate          → single text + quality 4-factor score
POST /api/translation/translate/batch    → list-of-sentences translation

=== Document Intelligence ===
GET  /api/documents/health     → document engine status
GET  /api/documents/templates  → List templates (nlp-report + 5 official JP templates)
GET  /api/documents/templates/{id} → Template field schema
POST /api/documents/generate/{id}          → metadata-only response
POST /api/documents/generate/{id}/download → StreamingResponse DOCX/PDF
POST /api/documents/generate/nlp-report            → Bilingual NLP report
POST /api/documents/generate/nlp-report/download   → Download NLP report
POST /api/documents/generate/company-analysis-report → Company analysis report
POST /api/documents/generate/company-analysis-report/from-dataset → from stored dataset
POST /api/documents/outline/draft       → AI-assisted outline drafting
POST /api/documents/batch/generate/download → Batch template generation
POST /api/documents/merge-pdfs          → Bilingual PDF merging
POST /api/documents/translate-docx      → DOCX translation
GET  /api/documents/{doc_id}/download   → Download by ledger ID
GET  /api/documents?limit=N             → Recent documents (SQLite ledger)

=== Machine Learning ===
GET  /api/ml/health            → engines + classes
GET  /api/ml/classifier/info   → classes / features / training_samples / calibrated
POST /api/ml/classify          → label + top_k probabilities (Sudachi TF-IDF + LinearSVC)
POST /api/ml/classify/batch    → batch classification
POST /api/ml/cluster           → KMeans clustering + top-keywords per cluster
GET  /api/ml/cluster/demo      → runs on built-in TRAIN_CATEGORIES corpus
POST /api/ml/topics            → LDA topic modeling
POST /api/ml/auto-label        → confidence-gated auto-labeling suggestions
GET  /api/ml/intent/info       → classes + rule_pattern counts
POST /api/ml/intent            → greeting/question/request/command/opinion/other (rule+ML)
POST /api/ml/intent/batch      → batch intent detection
GET  /api/ml/intent/evaluate   → intent evaluation metrics
GET  /api/ml/torch/info        → PyTorch MLP classifier status
POST /api/ml/torch/train       → Train PyTorch MLP classifier
POST /api/ml/torch/classify    → PyTorch MLP classification
GET  /api/ml/torch-transformer/info   → Transformer classifier status
POST /api/ml/torch-transformer/train  → Train Transformer classifier
POST /api/ml/torch-transformer/classify → Transformer classification
        """, language="shell")

    with st.expander("Development Principles"):
        st.markdown("""
            - AI / NLP functionality is the project core (not a simple CRUD management app)
            - Custom models trained for suitable ML/NLP tasks (not only API wrappers)
            - Streamlit is the main UI; Gradio is used only for specialized audio / voice AI
            - No heavy AI models downloaded during foundation setup (Whisper models are cached on first STT run)
            - Modular, readable code with meaningful naming
            - All DB passwords via `.env` / `python-dotenv` — no hardcoding
        """)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    _reiatsu_background()
    with st.sidebar:
        st.markdown(
            '<div class="soul-brand">'
            '<div class="soul-brand-mark">要</div>'
            '<div>'
            '<div class="soul-brand-name">KANAME</div>'
            '<div class="soul-brand-sub">Soul Reaper Console</div>'
            '</div>'
            '</div>'
            '<hr class="soul-divider">',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="sidebar-section">Navigation</div>', unsafe_allow_html=True)

        pages = {
            "Dashboard": render_dashboard,
            "Japanese NLP Intelligence": render_nlp,
            "Voice Intelligence": render_voice,
            "Data Intelligence": render_data,
            "Translation Intelligence": render_translation,
            "Document Intelligence": render_document,
            "Machine Learning": render_ml,
            "Project Information": render_project_info,
        }

        selection = st.radio("Modules", list(pages.keys()), label_visibility="collapsed")

        st.markdown('<hr class="soul-divider">', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-section">System Status</div>', unsafe_allow_html=True)
        online, s, lat = backend_online()
        cls = "bi-online" if online else "bi-offline"
        dot = "●"
        lat_txt = f" · {int(lat * 1000)} ms" if online else ""
        st.markdown(
            f'<div class="backend-indicator {cls}">{dot} Backend {s}{lat_txt}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="soul-footer">Version <b>0.4.0</b> &middot; Phase 4<br>'
            '<b>7</b> Modules Implemented</div>',
            unsafe_allow_html=True,
        )

    pages[selection]()


if __name__ == "__main__":
    main()
