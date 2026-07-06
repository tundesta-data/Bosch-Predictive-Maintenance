import streamlit as st
import pandas as pd
import sqlalchemy
import os
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# ── Page config ─────────────────────────────────────────────
st.set_page_config(
    page_title="Bosch Rexroth — Predictive Maintenance",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom styling ───────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0D1117; }
    
    /* Metric cards */
    .metric-card {
        background: #161B22;
        border: 1px solid #30363D;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        margin-bottom: 10px;
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #8B949E;
        margin-top: 4px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Risk badges */
    .badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.04em;
    }
    .badge-critical  { background: #3D1A1A; color: #FF6B6B; border: 1px solid #FF6B6B; }
    .badge-high      { background: #3D2B1A; color: #FFA94D; border: 1px solid #FFA94D; }
    .badge-elevated  { background: #3D361A; color: #FFD43B; border: 1px solid #FFD43B; }
    .badge-moderate  { background: #1A2B3D; color: #74C0FC; border: 1px solid #74C0FC; }
    .badge-healthy   { background: #1A3D2B; color: #69DB7C; border: 1px solid #69DB7C; }
    
    /* Machine card */
    .machine-card {
        background: #161B22;
        border: 1px solid #30363D;
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 8px;
        cursor: pointer;
        transition: border-color 0.2s;
    }
    .machine-card:hover { border-color: #58A6FF; }
    
    /* Section headers */
    .section-header {
        font-size: 0.75rem;
        font-weight: 600;
        color: #8B949E;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 12px;
        padding-bottom: 6px;
        border-bottom: 1px solid #30363D;
    }

    /* Hide default streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
</style>
""", unsafe_allow_html=True)

# ── Database connection ──────────────────────────────────────
@st.cache_resource
def get_engine():
    db_url = st.secrets.get("DATABASE_URL") or os.environ.get("DATABASE_URL")
    if not db_url:
        st.error("DATABASE_URL not configured. Add it to Streamlit secrets.")
        st.stop()
    return sqlalchemy.create_engine(db_url)

@st.cache_data(ttl=300)  # refresh every 5 minutes
def load_predictions():
    engine = get_engine()
    df = pd.read_sql("SELECT * FROM ml_predictions ORDER BY predicted_rul_hours ASC", engine)
    return df

# ── Risk colour helpers ──────────────────────────────────────
RISK_COLOURS = {
    "CRITICAL":  "#FF6B6B",
    "HIGH RISK": "#FFA94D",
    "ELEVATED":  "#FFD43B",
    "MODERATE":  "#74C0FC",
    "HEALTHY":   "#69DB7C",
}
BADGE_CLASS = {
    "CRITICAL":  "badge-critical",
    "HIGH RISK": "badge-high",
    "ELEVATED":  "badge-elevated",
    "MODERATE":  "badge-moderate",
    "HEALTHY":   "badge-healthy",
}

def risk_badge(risk):
    cls = BADGE_CLASS.get(risk, "badge-moderate")
    return f'<span class="badge {cls}">{risk}</span>'

# ── Load data ────────────────────────────────────────────────
try:
    df = load_predictions()
except Exception as e:
    st.error(f"Could not load predictions: {e}")
    st.stop()

if df.empty:
    st.warning("No predictions yet — check back after the next scheduled run.")
    st.stop()

# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Bosch Rexroth")
    st.markdown("**Predictive Maintenance**")
    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["🏭 Fleet Overview", "🔍 Machine Detail"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown(
        f"<div style='font-size:0.75rem; color:#8B949E;'>"
        f"Last prediction run:<br>"
        f"<b style='color:#58A6FF;'>"
        f"{pd.to_datetime(df['prediction_generated_at'].max()).strftime('%d %b %Y, %H:%M')}"
        f"</b></div>",
        unsafe_allow_html=True
    )
    st.markdown(
        f"<div style='font-size:0.75rem; color:#8B949E; margin-top:8px;'>"
        f"Machines monitored: <b style='color:#58A6FF;'>{len(df)}</b></div>",
        unsafe_allow_html=True
    )

    if st.button("🔄 Refresh data"):
        st.cache_data.clear()
        st.rerun()

# ════════════════════════════════════════════════════════════
# PAGE 1 — FLEET OVERVIEW
# ════════════════════════════════════════════════════════════
if page == "🏭 Fleet Overview":

    st.markdown("# Fleet Health Overview")
    st.markdown(
        "<p style='color:#8B949E;'>Live predictions refreshed every hour by the cloud pipeline.</p>",
        unsafe_allow_html=True
    )

    # ── Summary metrics ──────────────────────────────────────
    critical  = len(df[df['risk_level'] == 'CRITICAL'])
    high_risk = len(df[df['risk_level'] == 'HIGH RISK'])
    elevated  = len(df[df['risk_level'] == 'ELEVATED'])
    healthy   = len(df[df['risk_level'].isin(['MODERATE', 'HEALTHY'])])
    avg_rul   = df['predicted_rul_hours'].mean()

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-value" style="color:#FF6B6B;">{critical}</p>
            <p class="metric-label">🔴 Critical</p>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-value" style="color:#FFA94D;">{high_risk}</p>
            <p class="metric-label">🟠 High Risk</p>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-value" style="color:#FFD43B;">{elevated}</p>
            <p class="metric-label">🟡 Elevated</p>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-value" style="color:#69DB7C;">{healthy}</p>
            <p class="metric-label">🟢 Healthy</p>
        </div>""", unsafe_allow_html=True)
    with col5:
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-value" style="color:#58A6FF;">{avg_rul:.0f}h</p>
            <p class="metric-label">Avg RUL</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Charts row ───────────────────────────────────────────
    col_left, col_right = st.columns([1.2, 1])

    with col_left:
        st.markdown('<div class="section-header">Remaining Useful Life by Machine</div>',
                    unsafe_allow_html=True)
        bar_colours = [RISK_COLOURS.get(r, "#58A6FF") for r in df['risk_level']]
        fig = go.Figure(go.Bar(
            x=df['machine_id'],
            y=df['predicted_rul_hours'],
            marker_color=bar_colours,
            text=df['predicted_rul_hours'].round(0).astype(int).astype(str) + 'h',
            textposition='outside',
        ))
        fig.update_layout(
            paper_bgcolor='#161B22',
            plot_bgcolor='#161B22',
            font_color='#C9D1D9',
            margin=dict(t=10, b=10, l=0, r=0),
            height=280,
            xaxis=dict(gridcolor='#30363D'),
            yaxis=dict(gridcolor='#30363D', title='Hours'),
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.markdown('<div class="section-header">Fleet Risk Distribution</div>',
                    unsafe_allow_html=True)
        risk_counts = df['risk_level'].value_counts().reset_index()
        risk_counts.columns = ['Risk Level', 'Count']
        colours = [RISK_COLOURS.get(r, "#58A6FF") for r in risk_counts['Risk Level']]
        fig2 = go.Figure(go.Pie(
            labels=risk_counts['Risk Level'],
            values=risk_counts['Count'],
            marker_colors=colours,
            hole=0.55,
            textinfo='label+value',
            textfont_size=12,
        ))
        fig2.update_layout(
            paper_bgcolor='#161B22',
            plot_bgcolor='#161B22',
            font_color='#C9D1D9',
            margin=dict(t=10, b=10, l=0, r=0),
            height=280,
            showlegend=False
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ── Alerts table ─────────────────────────────────────────
    st.markdown('<div class="section-header">All Machines — Prediction Summary</div>',
                unsafe_allow_html=True)

    display_df = df[[
        'machine_id', 'risk_level', 'predicted_rul_hours',
        'shutdown_countdown', 'recommended_action',
        'predicted_failure_mode', 'failure_probability_pct',
        'pressure_bar', 'temp_celsius', 'vibration_x_g'
    ]].copy()

    display_df.columns = [
        'Machine', 'Risk Level', 'RUL (hrs)',
        'Shutdown In', 'Action',
        'Failure Mode', 'Failure Prob %',
        'Pressure (bar)', 'Temp (°C)', 'Vibration (g)'
    ]
    display_df['RUL (hrs)'] = display_df['RUL (hrs)'].round(1)
    display_df['Failure Prob %'] = display_df['Failure Prob %'].round(1)
    display_df['Pressure (bar)'] = display_df['Pressure (bar)'].round(1)
    display_df['Temp (°C)'] = display_df['Temp (°C)'].round(1)
    display_df['Vibration (g)'] = display_df['Vibration (g)'].round(3)

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        height=350
    )

# ════════════════════════════════════════════════════════════
# PAGE 2 — MACHINE DETAIL
# ════════════════════════════════════════════════════════════
elif page == "🔍 Machine Detail":

    st.markdown("# Machine Detail")

    machine_ids = df['machine_id'].tolist()
    selected = st.selectbox(
        "Select a machine to inspect",
        machine_ids,
        index=0
    )

    row = df[df['machine_id'] == selected].iloc[0]

    # ── Header strip ─────────────────────────────────────────
    col_id, col_risk, col_action = st.columns([1, 1, 2])
    with col_id:
        st.markdown(
            f"<h2 style='margin:0; color:#C9D1D9;'>{row['machine_id']}</h2>",
            unsafe_allow_html=True
        )
    with col_risk:
        st.markdown(
            f"<div style='padding-top:8px;'>{risk_badge(row['risk_level'])}</div>",
            unsafe_allow_html=True
        )
    with col_action:
        st.markdown(
            f"<div style='padding-top:8px; font-size:0.9rem; color:#8B949E;'>"
            f"Recommended action: <b style='color:#C9D1D9;'>{row['recommended_action']}</b></div>",
            unsafe_allow_html=True
        )

    st.markdown("---")

    # ── Key metrics ──────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        colour = RISK_COLOURS.get(row['risk_level'], '#58A6FF')
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-value" style="color:{colour};">{row['predicted_rul_hours']:.0f}h</p>
            <p class="metric-label">Remaining Useful Life</p>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-value" style="color:#C9D1D9; font-size:1.5rem;">{row['shutdown_countdown']}</p>
            <p class="metric-label">Shutdown Countdown</p>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-value" style="color:#FFA94D; font-size:1.4rem;">{row['predicted_failure_mode']}</p>
            <p class="metric-label">Predicted Failure Mode</p>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-value" style="color:#FF6B6B;">{row['failure_probability_pct']:.1f}%</p>
            <p class="metric-label">Failure Probability</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Sensor readings ──────────────────────────────────────
    st.markdown('<div class="section-header">Current Sensor Readings</div>',
                unsafe_allow_html=True)

    s1, s2, s3, s4, s5 = st.columns(5)
    sensors = [
        (s1, "Pressure", f"{row['pressure_bar']:.1f} bar",
         "#FF6B6B" if row['pressure_bar'] > 130 else "#69DB7C"),
        (s2, "Temperature", f"{row['temp_celsius']:.1f} °C",
         "#FF6B6B" if row['temp_celsius'] > 80 else "#69DB7C"),
        (s3, "Vibration X", f"{row['vibration_x_g']:.3f} g",
         "#FF6B6B" if row['vibration_x_g'] > 0.8 else "#69DB7C"),
        (s4, "Flow Rate", f"{row['flow_lpm']:.1f} L/min",
         "#FF6B6B" if row['flow_lpm'] < 60 else "#69DB7C"),
        (s5, "Pump RPM", f"{row['pump_rpm']:.0f} RPM", "#58A6FF"),
    ]
    for col, label, value, colour in sensors:
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <p class="metric-value" style="color:{colour}; font-size:1.4rem;">{value}</p>
                <p class="metric-label">{label}</p>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Danger indicators ────────────────────────────────────
    st.markdown('<div class="section-header">Danger Indicators</div>',
                unsafe_allow_html=True)

    danger_cols = st.columns(4)
    dangers = [
        ("Pressure Danger", row['pressure_bar'] > 130),
        ("Temp Danger",     row['temp_celsius'] > 80),
        ("Vibration Danger",row['vibration_x_g'] > 0.8),
        ("Flow Danger",     row['flow_lpm'] < 60),
    ]
    for i, (label, is_danger) in enumerate(dangers):
        with danger_cols[i]:
            icon = "🔴" if is_danger else "🟢"
            status = "DANGER" if is_danger else "NORMAL"
            colour = "#FF6B6B" if is_danger else "#69DB7C"
            st.markdown(f"""
            <div class="metric-card">
                <p style="font-size:1.5rem; margin:0;">{icon}</p>
                <p class="metric-value" style="color:{colour}; font-size:1rem;">{status}</p>
                <p class="metric-label">{label}</p>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Gauge chart — RUL ────────────────────────────────────
    st.markdown('<div class="section-header">RUL Gauge</div>',
                unsafe_allow_html=True)

    rul = float(row['predicted_rul_hours'])
    max_rul = 500
    fig3 = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=rul,
        delta={'reference': 168, 'valueformat': '.0f'},
        number={'suffix': ' hrs', 'font': {'color': '#C9D1D9'}},
        gauge={
            'axis': {'range': [0, max_rul], 'tickcolor': '#8B949E'},
            'bar': {'color': RISK_COLOURS.get(row['risk_level'], '#58A6FF')},
            'bgcolor': '#161B22',
            'steps': [
                {'range': [0, 24],   'color': '#3D1A1A'},
                {'range': [24, 48],  'color': '#3D2B1A'},
                {'range': [48, 72],  'color': '#3D361A'},
                {'range': [72, 168], 'color': '#1A2B3D'},
                {'range': [168, max_rul], 'color': '#1A3D2B'},
            ],
            'threshold': {
                'line': {'color': '#FF6B6B', 'width': 3},
                'thickness': 0.75,
                'value': 24
            }
        }
    ))
    fig3.update_layout(
        paper_bgcolor='#161B22',
        font_color='#C9D1D9',
        margin=dict(t=20, b=20, l=40, r=40),
        height=280
    )
    st.plotly_chart(fig3, use_container_width=True)

    # ── Footer ───────────────────────────────────────────────
    st.markdown(
        f"<p style='color:#8B949E; font-size:0.75rem; margin-top:20px;'>"
        f"Prediction generated at: {pd.to_datetime(row['prediction_generated_at']).strftime('%d %b %Y, %H:%M UTC')}"
        f" · Urgency window: {row['urgency_window']}"
        f" · Shutdown action: {row['shutdown_action']}</p>",
        unsafe_allow_html=True
    )
