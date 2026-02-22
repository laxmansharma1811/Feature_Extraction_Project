import streamlit as st
import joblib
import pandas as pd
import os
import time

# =============================================
#  CardioSense AI - Heart Disease Predictor
#  Designed & Developed by Laxman Sharma
# =============================================

# ----- Page Config -----
st.set_page_config(
    page_title="CardioSense AI | By Laxman Sharma",
    page_icon="https://em-content.zobj.net/source/apple/391/anatomical-heart_1fac0.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----- Premium Custom CSS -----
st.markdown("""
<style>
    /* ===== GOOGLE FONTS ===== */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

    /* ===== ROOT VARIABLES ===== */
    :root {
        --primary: #E63946;
        --primary-light: #FF6B6B;
        --primary-dark: #C1121F;
        --accent: #457B9D;
        --accent-light: #A8DADC;
        --bg-dark: #06080F;
        --bg-card: #0E1525;
        --bg-card-alt: #121A2E;
        --text-primary: #F1FAEE;
        --text-secondary: #7B8794;
        --text-muted: #4A5568;
        --border-color: #1A2440;
        --border-glow: rgba(230, 57, 70, 0.2);
        --success: #06D6A0;
        --success-dark: #04A77D;
        --warning: #FFD166;
        --danger: #EF476F;
        --glass-bg: rgba(14, 21, 37, 0.75);
        --glass-border: rgba(255, 255, 255, 0.06);
    }

    /* ===== GLOBAL ===== */
    .stApp {
        background: var(--bg-dark);
        background-image:
            radial-gradient(ellipse 80% 50% at 50% -25%, rgba(230, 57, 70, 0.12), transparent),
            radial-gradient(ellipse 60% 35% at 85% 70%, rgba(69, 123, 157, 0.07), transparent);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: var(--text-primary);
    }

    /* Hide Streamlit chrome */
    #MainMenu, footer, header, .stDeployButton { display: none !important; visibility: hidden !important; }

    /* ===== SCROLLBAR ===== */
    ::-webkit-scrollbar { width: 5px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: var(--primary); border-radius: 5px; }
    ::-webkit-scrollbar-thumb:hover { background: var(--primary-light); }

    /* ===== SIDEBAR ===== */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #080C18 0%, #0E1525 50%, #080C18 100%);
        border-right: 1px solid var(--glass-border);
    }
    section[data-testid="stSidebar"] .stMarkdown p {
        color: var(--text-secondary);
        font-size: 0.88rem;
        line-height: 1.7;
    }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: var(--text-primary) !important;
        font-family: 'Space Grotesk', sans-serif;
    }

    /* ===== MAIN BLOCK ===== */
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 2rem !important;
        max-width: 1200px;
    }

    /* ===== TYPOGRAPHY ===== */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Space Grotesk', sans-serif !important;
        color: var(--text-primary) !important;
    }
    h1 { font-size: 2.5rem !important; font-weight: 800 !important; letter-spacing: -0.03em; }

    /* ===== HERO BANNER ===== */
    .hero-banner {
        background: linear-gradient(135deg, rgba(230, 57, 70, 0.1) 0%, rgba(69, 123, 157, 0.06) 100%);
        border: 1px solid var(--glass-border);
        border-radius: 24px;
        padding: 3rem 2.5rem;
        margin-bottom: 2rem;
        backdrop-filter: blur(24px);
        position: relative;
        overflow: hidden;
    }
    .hero-banner::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, var(--primary), var(--accent), var(--primary-light));
    }
    .hero-banner::after {
        content: '';
        position: absolute;
        top: -50%;
        right: -10%;
        width: 400px;
        height: 400px;
        background: radial-gradient(circle, rgba(230,57,70,0.08) 0%, transparent 70%);
        border-radius: 50%;
        pointer-events: none;
    }
    .hero-tag {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(230, 57, 70, 0.12);
        border: 1px solid rgba(230, 57, 70, 0.25);
        color: var(--primary-light);
        padding: 7px 16px;
        border-radius: 50px;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 1.2rem;
    }
    .hero-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #FFFFFF 0%, #E63946 60%, #FF6B6B 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        line-height: 1.1;
        margin-bottom: 0.6rem;
    }
    .hero-desc {
        font-size: 1.05rem;
        color: var(--text-secondary);
        line-height: 1.7;
        max-width: 600px;
    }

    /* ===== GLASS CARD ===== */
    .glass-card {
        background: var(--glass-bg);
        border: 1px solid var(--glass-border);
        border-radius: 18px;
        padding: 1.8rem;
        backdrop-filter: blur(20px);
        transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    .glass-card:hover {
        border-color: var(--border-glow);
        transform: translateY(-2px);
        box-shadow: 0 12px 40px rgba(230, 57, 70, 0.08);
    }

    /* ===== SECTION HEADER ===== */
    .section-hdr {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 1.2rem;
    }
    .section-ico {
        width: 44px; height: 44px;
        border-radius: 14px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.25rem;
        flex-shrink: 0;
    }
    .ico-red  { background: rgba(230,57,70,0.12); border: 1px solid rgba(230,57,70,0.2); }
    .ico-blue { background: rgba(69,123,157,0.12); border: 1px solid rgba(69,123,157,0.2); }
    .ico-grn  { background: rgba(6,214,160,0.12); border: 1px solid rgba(6,214,160,0.2); }
    .ico-gold { background: rgba(255,209,102,0.12); border: 1px solid rgba(255,209,102,0.2); }
    .section-lbl {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.1rem;
        font-weight: 700;
        color: var(--text-primary);
        margin: 0;
    }
    .section-sub {
        font-size: 0.78rem;
        color: var(--text-secondary);
        margin: 2px 0 0;
    }

    /* ===== INPUTS ===== */
    .stNumberInput > div > div > input,
    .stTextInput > div > div > input {
        background: rgba(26, 36, 64, 0.5) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 12px !important;
        color: var(--text-primary) !important;
        font-family: 'Inter', sans-serif !important;
        padding: 0.7rem 1rem !important;
        font-size: 0.95rem !important;
        transition: all 0.25s ease !important;
    }
    .stNumberInput > div > div > input:focus,
    .stTextInput > div > div > input:focus {
        border-color: var(--primary) !important;
        box-shadow: 0 0 0 3px rgba(230, 57, 70, 0.12) !important;
    }
    .stSelectbox > div > div {
        background: rgba(26, 36, 64, 0.5) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 12px !important;
        color: var(--text-primary) !important;
    }
    .stSelectbox > div > div:hover { border-color: var(--primary) !important; }
    .stSelectbox label, .stNumberInput label, .stTextInput label {
        color: var(--text-secondary) !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.83rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.02em;
    }

    /* ===== BUTTON ===== */
    .stButton > button {
        background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%) !important;
        color: #fff !important;
        border: none !important;
        border-radius: 16px !important;
        padding: 0.9rem 2.5rem !important;
        font-family: 'Space Grotesk', sans-serif !important;
        font-size: 1.05rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.04em;
        width: 100% !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 6px 24px rgba(230, 57, 70, 0.35) !important;
        text-transform: uppercase;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 10px 36px rgba(230, 57, 70, 0.5) !important;
        background: linear-gradient(135deg, var(--primary-light) 0%, var(--primary) 100%) !important;
    }
    .stButton > button:active { transform: translateY(0) !important; }

    /* ===== RESULT CARDS ===== */
    .result-card {
        border-radius: 22px;
        padding: 2.5rem 2rem;
        margin-top: 1rem;
        backdrop-filter: blur(24px);
        position: relative;
        overflow: hidden;
    }
    .result-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 4px;
        border-radius: 22px 22px 0 0;
    }
    .result-danger {
        background: linear-gradient(145deg, rgba(239,71,111,0.10) 0%, rgba(230,57,70,0.05) 100%);
        border: 1px solid rgba(239,71,111,0.25);
    }
    .result-danger::before { background: linear-gradient(90deg, #EF476F, #E63946, #FF6B6B); }
    .result-success {
        background: linear-gradient(145deg, rgba(6,214,160,0.10) 0%, rgba(69,123,157,0.05) 100%);
        border: 1px solid rgba(6,214,160,0.25);
    }
    .result-success::before { background: linear-gradient(90deg, #06D6A0, #457B9D, #A8DADC); }

    .result-ico { font-size: 3.5rem; margin-bottom: 0.6rem; }
    .result-ttl {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.7rem;
        font-weight: 700;
        margin-bottom: 0.4rem;
    }
    .result-danger .result-ttl { color: #EF476F; }
    .result-success .result-ttl { color: #06D6A0; }
    .result-prob {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 3.2rem;
        font-weight: 800;
        line-height: 1;
        margin: 0.4rem 0;
    }
    .result-danger .result-prob { color: #FF6B6B; }
    .result-success .result-prob { color: #06D6A0; }
    .result-lbl {
        font-size: 0.82rem;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-weight: 700;
    }

    /* ===== RISK BAR ===== */
    .risk-bar-bg {
        background: rgba(30, 41, 59, 0.5);
        border-radius: 10px;
        height: 14px;
        overflow: hidden;
        margin: 1rem 0;
    }
    .risk-bar-fill {
        height: 100%;
        border-radius: 10px;
        transition: width 1.2s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .bar-danger { background: linear-gradient(90deg, #EF476F, #E63946, #FF6B6B); }
    .bar-success { background: linear-gradient(90deg, #06D6A0, #457B9D, #A8DADC); }

    /* ===== METRIC CARD ===== */
    .metric-box {
        background: rgba(14, 21, 37, 0.6);
        border: 1px solid var(--glass-border);
        border-radius: 16px;
        padding: 1.3rem;
        text-align: center;
        transition: all 0.3s ease;
    }
    .metric-box:hover {
        border-color: var(--border-glow);
        background: rgba(14, 21, 37, 0.9);
    }
    .metric-val {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.9rem;
        font-weight: 800;
        color: var(--text-primary);
        line-height: 1;
    }
    .metric-lbl {
        font-size: 0.72rem;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 700;
        margin-top: 0.4rem;
    }

    /* ===== STAT ROW ===== */
    .stat-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.65rem 0;
        border-bottom: 1px solid rgba(255,255,255,0.03);
    }
    .stat-row:last-child { border-bottom: none; }
    .stat-key { font-size: 0.85rem; color: var(--text-secondary); }
    .stat-val {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 600;
        color: var(--text-primary);
        font-size: 0.95rem;
    }

    /* ===== DIVIDER ===== */
    .divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, var(--glass-border), transparent);
        margin: 1.5rem 0;
        border: none;
    }

    /* ===== INFO / DISCLAIMER ===== */
    .info-box {
        background: rgba(69,123,157,0.08);
        border: 1px solid rgba(69,123,157,0.18);
        border-radius: 14px;
        padding: 1.1rem 1.4rem;
        font-size: 0.85rem;
        color: var(--accent-light);
        line-height: 1.7;
    }
    .disclaimer-box {
        background: rgba(255,209,102,0.06);
        border: 1px solid rgba(255,209,102,0.15);
        border-radius: 14px;
        padding: 1.1rem 1.4rem;
        font-size: 0.82rem;
        color: var(--warning);
        margin-top: 1.5rem;
        text-align: center;
    }

    /* ===== FOOTER ===== */
    .app-footer {
        text-align: center;
        padding: 2.5rem 0 1rem;
        margin-top: 3rem;
        border-top: 1px solid var(--glass-border);
    }
    .footer-name {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.4rem;
        font-weight: 800;
        background: linear-gradient(135deg, var(--primary-light), var(--accent-light));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .footer-tag {
        font-size: 0.8rem;
        color: var(--text-muted);
        margin-top: 0.3rem;
    }

    /* ===== ANIMATIONS ===== */
    @keyframes pulseGlow {
        0%, 100% { box-shadow: 0 0 20px rgba(230,57,70,0.08); }
        50% { box-shadow: 0 0 50px rgba(230,57,70,0.2); }
    }
    .pulse-glow { animation: pulseGlow 3s ease-in-out infinite; }

    @keyframes slideUp {
        from { opacity: 0; transform: translateY(24px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .slide-up { animation: slideUp 0.5s ease-out forwards; }

    @keyframes heartbeat {
        0%, 100% { transform: scale(1); }
        14% { transform: scale(1.15); }
        28% { transform: scale(1); }
        42% { transform: scale(1.1); }
        56% { transform: scale(1); }
    }
    .heartbeat { animation: heartbeat 2s ease-in-out infinite; }

    /* ===== TABS ===== */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: rgba(14, 21, 37, 0.6);
        border-radius: 16px;
        padding: 5px;
        border: 1px solid var(--glass-border);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 12px;
        color: var(--text-secondary);
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        padding: 0.65rem 1.4rem;
        font-size: 0.9rem;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(230,57,70,0.12) !important;
        color: var(--primary-light) !important;
        border-bottom: none !important;
    }
    .stTabs [data-baseweb="tab-highlight"],
    .stTabs [data-baseweb="tab-border"] { display: none; }

    /* ===== EXPANDER ===== */
    .streamlit-expanderHeader {
        background: rgba(14,21,37,0.6) !important;
        border: 1px solid var(--glass-border) !important;
        border-radius: 12px !important;
        color: var(--text-primary) !important;
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 600 !important;
    }

    /* ===== ALERTS ===== */
    .stAlert > div {
        border-radius: 14px !important;
        border: none !important;
        font-family: 'Inter', sans-serif !important;
    }

    /* Hide anchor links */
    h1 a, h2 a, h3 a { display: none !important; }

    /* ===== RESPONSIVE ===== */
    @media (max-width: 768px) {
        .hero-title { font-size: 2.1rem !important; }
        .hero-desc { font-size: 0.92rem; }
        .block-container { padding-left: 0.8rem !important; padding-right: 0.8rem !important; }
        .glass-card { padding: 1.2rem; }
        .result-card { padding: 1.5rem; }
        .result-prob { font-size: 2.4rem; }
        h1 { font-size: 1.8rem !important; }
        .hero-banner { padding: 2rem 1.5rem; }
    }
    @media (max-width: 480px) {
        .hero-title { font-size: 1.6rem !important; }
        .hero-banner { padding: 1.5rem 1rem; border-radius: 16px; }
        .result-card { padding: 1.2rem; border-radius: 16px; }
        .result-prob { font-size: 2rem; }
        .metric-box { padding: 0.9rem; }
        .metric-val { font-size: 1.4rem; }
    }
</style>
""", unsafe_allow_html=True)


# =============================================
#  SIDEBAR
# =============================================
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 1.2rem 0 0.8rem;">
        <div class="heartbeat" style="font-size: 3.5rem; margin-bottom: 0.6rem;">&#129728;</div>
        <div style="font-family: 'Space Grotesk', sans-serif; font-size: 1.5rem; font-weight: 800;
                    background: linear-gradient(135deg, #FFFFFF 20%, #E63946 80%);
                    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                    background-clip: text;">CardioSense AI</div>
        <div style="font-size: 0.78rem; color: #7B8794; margin-top: 0.4rem;
                    letter-spacing: 0.05em;">Intelligent Heart Health Analysis</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    st.markdown("### About")
    st.markdown("""
    ML-powered cardiac risk assessment that analyzes **11 clinical parameters** to predict heart disease likelihood with probability scores.

    **Technology Stack:**
    - Scikit-Learn ML Pipeline
    - Streamlit Framework
    - Custom Dark UI Theme
    """)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    st.markdown("### Clinical Reference")

    with st.expander("Chest Pain Types"):
        st.markdown("""
        | Code | Description |
        |------|-------------|
        | **TA** | Typical Angina |
        | **ATA** | Atypical Angina |
        | **NAP** | Non-Anginal Pain |
        | **ASY** | Asymptomatic |
        """)

    with st.expander("Resting ECG"):
        st.markdown("""
        | Code | Description |
        |------|-------------|
        | **Normal** | Normal ECG |
        | **ST** | ST-T wave abnormality |
        | **LVH** | Left ventricular hypertrophy |
        """)

    with st.expander("ST Slope"):
        st.markdown("""
        | Code | Description |
        |------|-------------|
        | **Up** | Upsloping |
        | **Flat** | Flat |
        | **Down** | Downsloping |
        """)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # Developer credit
    st.markdown("""
    <div style="text-align:center; padding: 0.8rem 0;">
        <div style="font-size: 0.7rem; color: #4A5568; text-transform: uppercase;
                    letter-spacing: 0.12em; font-weight: 700;">Designed & Developed by</div>
        <div style="font-family: 'Space Grotesk', sans-serif; font-size: 1.2rem; font-weight: 800;
                    background: linear-gradient(135deg, #FF6B6B, #A8DADC);
                    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                    background-clip: text; margin-top: 0.4rem;">Laxman Sharma</div>
        <div style="font-size: 0.68rem; color: #4A5568; margin-top: 0.3rem;">v2.0 &middot; 2025</div>
    </div>
    """, unsafe_allow_html=True)


# =============================================
#  HERO SECTION
# =============================================
st.markdown("""
<div class="hero-banner slide-up">
    <div class="hero-tag">
        <span>&#128300;</span> ML-Powered Cardiac Diagnostics
    </div>
    <div class="hero-title">Heart Disease<br/>Prediction System</div>
    <p class="hero-desc">
        Advanced AI-driven cardiac risk assessment. Enter patient clinical parameters
        for an instant analysis powered by machine learning.
    </p>
</div>
""", unsafe_allow_html=True)


# =============================================
#  LOAD MODEL
# =============================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "heart_pipeline.pkl")

model_loaded = False
model = None

if os.path.exists(model_path):
    try:
        model = joblib.load(model_path)
        model_loaded = True
    except Exception as e:
        st.error(f"Error loading model: {e}")
else:
    st.markdown("""
    <div class="info-box">
        <strong>&#9432; Setup Required:</strong> Place your trained model file
        <code style="background:rgba(230,57,70,0.15); padding:2px 8px; border-radius:6px; color:#FF6B6B;">heart_pipeline.pkl</code>
        in the same directory as this script for predictions to work.
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<br/>", unsafe_allow_html=True)


# =============================================
#  TABS: DATA ENTRY + RESULTS
# =============================================
tab_input, tab_result = st.tabs(["&#128203;  Patient Data Entry", "&#128202;  Prediction Results"])

# ----- TAB 1: INPUT -----
with tab_input:

    # --- Patient Demographics ---
    st.markdown("""
    <div class="section-hdr">
        <div class="section-ico ico-red">&#128100;</div>
        <div>
            <p class="section-lbl">Patient Demographics</p>
            <p class="section-sub">Basic patient information</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        Age = st.number_input("Age (years)", min_value=1, max_value=120, value=40,
                               help="Patient's age in years")
    with c2:
        Sex = st.selectbox("Biological Sex", ["M", "F"],
                            help="M = Male, F = Female")
    with c3:
        ChestPainType = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"],
                                      help="ATA: Atypical Angina | NAP: Non-Anginal | TA: Typical Angina | ASY: Asymptomatic")

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # --- Vital Signs ---
    st.markdown("""
    <div class="section-hdr">
        <div class="section-ico ico-blue">&#128137;</div>
        <div>
            <p class="section-lbl">Vital Signs & Blood Work</p>
            <p class="section-sub">Clinical measurements and lab results</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    c4, c5, c6 = st.columns(3)
    with c4:
        RestingBP = st.number_input("Resting BP (mm Hg)", min_value=0, max_value=300, value=120,
                                     help="Resting blood pressure in mm Hg")
    with c5:
        Cholesterol = st.number_input("Cholesterol (mg/dl)", min_value=0, max_value=600, value=200,
                                       help="Serum cholesterol in mg/dl")
    with c6:
        FastingBS = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1],
                                  help="0 = No, 1 = Yes")

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # --- Cardiac Assessment ---
    st.markdown("""
    <div class="section-hdr">
        <div class="section-ico ico-grn">&#129728;</div>
        <div>
            <p class="section-lbl">Cardiac Assessment</p>
            <p class="section-sub">Heart function and exercise response</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    c7, c8 = st.columns(2)
    with c7:
        RestingECG = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"],
                                   help="Normal | ST-T wave abnormality | Left ventricular hypertrophy")
        ExerciseAngina = st.selectbox("Exercise Induced Angina", ["N", "Y"],
                                       help="Y = Yes, N = No")
    with c8:
        MaxHR = st.number_input("Max Heart Rate (bpm)", min_value=50, max_value=250, value=150,
                                 help="Maximum heart rate achieved during exercise")
        Oldpeak = st.number_input("Oldpeak (ST Depression)", min_value=-5.0, max_value=10.0, value=1.0,
                                   step=0.1, help="ST depression induced by exercise relative to rest")

    ST_Slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"],
                             help="Up: Upsloping | Flat | Down: Downsloping")

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # --- Input Summary ---
    st.markdown("""
    <div class="section-hdr">
        <div class="section-ico ico-gold">&#128203;</div>
        <div>
            <p class="section-lbl">Quick Summary</p>
            <p class="section-sub">Review your entries before analysis</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    for col, (lbl, val, unit) in zip(
        [m1, m2, m3, m4],
        [("Age", str(Age), "yrs"), ("Resting BP", str(RestingBP), "mmHg"),
         ("Cholesterol", str(Cholesterol), "mg/dl"), ("Max HR", str(MaxHR), "bpm")]
    ):
        with col:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-val">{val}</div>
                <div class="metric-lbl">{lbl} ({unit})</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)

    m5, m6, m7, m8 = st.columns(4)
    for col, (lbl, val) in zip(
        [m5, m6, m7, m8],
        [("Sex", Sex), ("Chest Pain", ChestPainType),
         ("ECG", RestingECG), ("Oldpeak", str(Oldpeak))]
    ):
        with col:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-val" style="font-size:1.5rem;">{val}</div>
                <div class="metric-lbl">{lbl}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)

    # --- Predict Button ---
    predict_clicked = st.button("&#129728;  Analyze Heart Health", use_container_width=True)


# ----- TAB 2: RESULTS -----
with tab_result:
    if predict_clicked and model_loaded:

        # Loading animation
        with st.spinner(""):
            loader = st.empty()
            loader.markdown("""
            <div style="text-align:center; padding: 3rem;">
                <div class="heartbeat" style="font-size: 4rem; margin-bottom: 1rem;">&#129728;</div>
                <div style="font-family: 'Space Grotesk', sans-serif; font-size: 1.2rem;
                            color: #F1FAEE; font-weight: 700;">Analyzing cardiac parameters...</div>
                <div style="font-size: 0.85rem; color: #7B8794; margin-top: 0.5rem;">
                    Running ML model inference</div>
            </div>
            """, unsafe_allow_html=True)
            time.sleep(1.5)
            loader.empty()

        # Build input & predict
        input_data = {
            "Age": Age, "Sex": Sex, "ChestPainType": ChestPainType,
            "RestingBP": RestingBP, "Cholesterol": Cholesterol, "FastingBS": FastingBS,
            "RestingECG": RestingECG, "MaxHR": MaxHR, "ExerciseAngina": ExerciseAngina,
            "Oldpeak": Oldpeak, "ST_Slope": ST_Slope
        }
        input_df = pd.DataFrame([input_data])
        prediction = model.predict(input_df)
        probability = model.predict_proba(input_df)

        risk_pct = probability[0][1] * 100
        safe_pct = probability[0][0] * 100

        # --- HIGH RISK ---
        if prediction[0] == 1:
            st.markdown(f"""
            <div class="result-card result-danger slide-up pulse-glow">
                <div style="display: flex; align-items: flex-start; gap: 2rem; flex-wrap: wrap;">
                    <div style="flex: 1; min-width: 260px;">
                        <div class="result-ico">&#128680;</div>
                        <div class="result-ttl">High Risk Detected</div>
                        <p style="color: #7B8794; font-size: 0.9rem; line-height: 1.7; margin-bottom: 1rem;">
                            The AI model has identified elevated cardiac risk markers.
                            Immediate consultation with a cardiologist is strongly recommended.
                        </p>
                        <div class="result-lbl">Risk Probability</div>
                        <div class="result-prob">{risk_pct:.1f}%</div>
                        <div class="risk-bar-bg">
                            <div class="risk-bar-fill bar-danger" style="width: {risk_pct}%;"></div>
                        </div>
                    </div>
                    <div style="flex: 0 0 210px; min-width: 180px;">
                        <div class="metric-box" style="margin-bottom: 0.8rem; border-color: rgba(239,71,111,0.2);">
                            <div class="metric-val" style="color: #EF476F;">{risk_pct:.1f}%</div>
                            <div class="metric-lbl">Disease Risk</div>
                        </div>
                        <div class="metric-box" style="border-color: rgba(69,123,157,0.2);">
                            <div class="metric-val" style="color: #A8DADC;">{safe_pct:.1f}%</div>
                            <div class="metric-lbl">Safe Margin</div>
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # --- LOW RISK ---
        else:
            st.markdown(f"""
            <div class="result-card result-success slide-up">
                <div style="display: flex; align-items: flex-start; gap: 2rem; flex-wrap: wrap;">
                    <div style="flex: 1; min-width: 260px;">
                        <div class="result-ico">&#9989;</div>
                        <div class="result-ttl">Low Risk Assessment</div>
                        <p style="color: #7B8794; font-size: 0.9rem; line-height: 1.7; margin-bottom: 1rem;">
                            Based on the clinical parameters provided, the AI model indicates a low
                            probability of heart disease. Continue maintaining a healthy lifestyle.
                        </p>
                        <div class="result-lbl">Confidence Level</div>
                        <div class="result-prob">{safe_pct:.1f}%</div>
                        <div class="risk-bar-bg">
                            <div class="risk-bar-fill bar-success" style="width: {safe_pct}%;"></div>
                        </div>
                    </div>
                    <div style="flex: 0 0 210px; min-width: 180px;">
                        <div class="metric-box" style="margin-bottom: 0.8rem; border-color: rgba(6,214,160,0.2);">
                            <div class="metric-val" style="color: #06D6A0;">{safe_pct:.1f}%</div>
                            <div class="metric-lbl">Confidence</div>
                        </div>
                        <div class="metric-box" style="border-color: rgba(239,71,111,0.15);">
                            <div class="metric-val" style="color: #FF6B6B;">{risk_pct:.1f}%</div>
                            <div class="metric-lbl">Risk Level</div>
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # --- Patient Data Summary ---
        st.markdown("<br/>", unsafe_allow_html=True)
        st.markdown("""
        <div class="section-hdr">
            <div class="section-ico ico-blue">&#128202;</div>
            <div>
                <p class="section-lbl">Patient Data Summary</p>
                <p class="section-sub">All input parameters used for analysis</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="glass-card">', unsafe_allow_html=True)

        dl, dr = st.columns(2)
        left_items = [
            ("Age", f"{Age} years"),
            ("Sex", "Male" if Sex == "M" else "Female"),
            ("Chest Pain Type", ChestPainType),
            ("Resting BP", f"{RestingBP} mm Hg"),
            ("Cholesterol", f"{Cholesterol} mg/dl"),
            ("Fasting Blood Sugar", "Yes (>120)" if FastingBS == 1 else "No"),
        ]
        right_items = [
            ("Resting ECG", RestingECG),
            ("Max Heart Rate", f"{MaxHR} bpm"),
            ("Exercise Angina", "Yes" if ExerciseAngina == "Y" else "No"),
            ("Oldpeak", str(Oldpeak)),
            ("ST Slope", ST_Slope),
            ("Prediction", "High Risk" if prediction[0] == 1 else "Low Risk"),
        ]

        with dl:
            for k, v in left_items:
                st.markdown(f'<div class="stat-row"><span class="stat-key">{k}</span><span class="stat-val">{v}</span></div>', unsafe_allow_html=True)
        with dr:
            for k, v in right_items:
                color = "#EF476F" if k == "Prediction" and v == "High Risk" else ("#06D6A0" if k == "Prediction" else "var(--text-primary)")
                st.markdown(f'<div class="stat-row"><span class="stat-key">{k}</span><span class="stat-val" style="color:{color};">{v}</span></div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

        # Disclaimer
        st.markdown("""
        <div class="disclaimer-box">
            <strong>Medical Disclaimer:</strong> This AI-powered tool is for educational and informational
            purposes only. It is NOT a substitute for professional medical advice, diagnosis, or treatment.
            Always consult a qualified healthcare provider.
        </div>
        """, unsafe_allow_html=True)

    elif predict_clicked and not model_loaded:
        st.markdown("""
        <div class="result-card result-danger slide-up" style="text-align:center;">
            <div class="result-ico">&#9888;&#65039;</div>
            <div class="result-ttl">Model Not Available</div>
            <p style="color: #7B8794; font-size: 0.9rem;">
                The prediction model (<code style="background:rgba(230,57,70,0.15); padding:2px 8px;
                border-radius:6px; color:#FF6B6B;">heart_pipeline.pkl</code>) could not be loaded.<br/>
                Place it in the same directory as this script.
            </p>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style="text-align:center; padding: 5rem 2rem;" class="slide-up">
            <div class="heartbeat" style="font-size: 5rem; margin-bottom: 1.2rem; opacity: 0.4;">&#129728;</div>
            <div style="font-family: 'Space Grotesk', sans-serif; font-size: 1.4rem;
                        color: #F1FAEE; font-weight: 700; margin-bottom: 0.5rem;">
                Awaiting Patient Data
            </div>
            <p style="color: #7B8794; font-size: 0.9rem; max-width: 420px; margin: 0 auto; line-height: 1.7;">
                Enter the patient's clinical parameters in the <strong style="color:#F1FAEE;">Patient Data Entry</strong> tab
                and click <strong style="color:#F1FAEE;">Analyze Heart Health</strong> to see results.
            </p>
        </div>
        """, unsafe_allow_html=True)


# =============================================
#  FOOTER
# =============================================
st.markdown("""
<div class="app-footer slide-up">
    <div style="font-size: 0.7rem; color: #4A5568; text-transform: uppercase;
                letter-spacing: 0.14em; font-weight: 700; margin-bottom: 0.4rem;">
        Designed & Developed with &#10084;&#65039; by
    </div>
    <div class="footer-name">Laxman Sharma</div>
    <div class="footer-tag">CardioSense AI v2.0 &middot; ML-Powered Heart Disease Prediction</div>
    <div style="margin-top: 0.8rem; font-size: 0.68rem; color: #3A4255;">
        Built with Streamlit &middot; Python &middot; Scikit-Learn
    </div>
</div>
""", unsafe_allow_html=True)
