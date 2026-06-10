#TARDIS — SNCF Delay Prediction Platform          
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import warnings
warnings.filterwarnings("ignore")

#  Page configuration
st.set_page_config(
    page_title="TARDIS · SNCF Analytics",
    page_icon="🚄",
    layout="wide",
    initial_sidebar_state="expanded",
)

#  THEME & CSS CUSTOM
st.markdown("""
<style>
/* ── Global ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    border-right: 1px solid #334155;
}
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label { color: #94a3b8 !important; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.05em; }

/* ── KPI Cards ── */
.kpi-card {
    background: linear-gradient(135deg, #1e293b, #0f172a);
    border: 1px solid #334155;
    border-radius: 16px;
    padding: 20px 24px;
    text-align: center;
    transition: transform .2s, box-shadow .2s;
}
.kpi-card:hover { transform: translateY(-2px); box-shadow: 0 8px 32px rgba(0,0,0,.4); }
.kpi-value { font-size: 2rem; font-weight: 700; color: #38bdf8; margin: 4px 0; }
.kpi-label { font-size: 0.75rem; color: #94a3b8; text-transform: uppercase; letter-spacing: .06em; }
.kpi-delta { font-size: 0.85rem; margin-top: 4px; }
.delta-up   { color: #f87171; }
.delta-down { color: #4ade80; }

/* ── Section headers ── */
.section-header {
    font-size: 1.05rem;
    font-weight: 600;
    color: #f1f5f9;
    border-left: 3px solid #38bdf8;
    padding-left: 12px;
    margin: 32px 0 16px;
}

/* ── Prediction result badge ── */
.pred-badge {
    border-radius: 12px;
    padding: 18px 24px;
    text-align: center;
    font-size: 1.6rem;
    font-weight: 700;
    letter-spacing: -.01em;
    margin-top: 16px;
}
.badge-green  { background: linear-gradient(135deg,#064e3b,#065f46); color:#6ee7b7; border: 1px solid #059669; }
.badge-orange { background: linear-gradient(135deg,#78350f,#92400e); color:#fcd34d; border: 1px solid #d97706; }
.badge-red    { background: linear-gradient(135deg,#7f1d1d,#991b1b); color:#fca5a5; border: 1px solid #dc2626; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: #0f172a;
    border-radius: 10px;
    padding: 4px;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    color: #94a3b8 !important;
    border-radius: 8px;
    padding: 8px 18px;
}
.stTabs [aria-selected="true"] {
    background: #1e40af !important;
    color: #fff !important;
}

/* ── Metrics override ── */
[data-testid="metric-container"] {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 12px 16px;
}

/* ── Dataframe ── */
.stDataFrame { border-radius: 10px; overflow: hidden; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0f172a; }
::-webkit-scrollbar-thumb { background: #334155; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

#  CONSTANTS & MAPS
MONTH_ORDER = [
    "January","February","March","April","May","June",
    "July","August","September","October","November","December"
]
DAY_ORDER = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
DAY_MAP   = {d: i for i, d in enumerate(DAY_ORDER)}
MONTH_MAP = {m: i+1 for i, m in enumerate(MONTH_ORDER)}

CAUSE_COLS = {
    "External causes":     "Pct delay due to external causes",
    "Infrastructure":      "Pct delay due to infrastructure",
    "Traffic management":  "Pct delay due to traffic management",
    "Rolling stock":       "Pct delay due to rolling stock",
    "Station management":  "Pct delay due to station management and equipment reuse",
    "Passenger handling":  "Pct delay due to passenger handling (crowding, disabled persons, connections)",
}

FEATURES = [
    "Departure station", "Arrival station",
    "Number of scheduled trains",
    "Number of cancelled trains",
    "Number of trains delayed at departure",
    "Average delay of late trains at departure",
    "Number of trains delayed > 15min",
    "Number of trains delayed > 30min",
    "Number of trains delayed > 60min",
    "Pct delay due to external causes",
    "Pct delay due to infrastructure",
    "Pct delay due to traffic management",
    "Pct delay due to rolling stock",
    "month", "year", "day_of_week",
]

PLOTLY_THEME = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter", color="#e2e8f0"),
    margin=dict(l=20, r=20, t=40, b=20),
)

BLUE_SEQ  = px.colors.sequential.Blues_r
ACCENT    = "#38bdf8"
RED_COLOR = "#f87171"
AMBER     = "#fbbf24"

#  DATA LOADING
@st.cache_data(show_spinner="Loading data...")
def load_data():
    df = pd.read_csv("cleaned_dataset.csv")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Months"] = pd.Categorical(df["Months"], categories=MONTH_ORDER, ordered=True)
    df["Days"]   = pd.Categorical(df["Days"],   categories=DAY_ORDER,   ordered=True)
    return df

@st.cache_resource(show_spinner="Loading model…")
def load_model():
    try:
        bundle = joblib.load("model.joblib")
        return bundle["model"], bundle["station_encoder"]
    except Exception:
        return None, None

df = load_data()
model, station_encoder = load_model()

#  SIDEBAR
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:20px 0 28px'>
      <div style='font-size:2.8rem'>🚄</div>
      <div style='font-size:1.4rem;font-weight:700;color:#f1f5f9'>TARDIS</div>
      <div style='font-size:.75rem;color:#64748b;letter-spacing:.1em'>SNCF DELAY INTELLIGENCE</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🔎 Global Filters")

    sel_service = st.multiselect(
        "Service type",
        options=df["Service"].unique().tolist(),
        default=df["Service"].unique().tolist(),
    )
    sel_years = st.select_slider(
        "Year range",
        options=sorted(df["Years"].unique()),
        value=(int(df["Years"].min()), int(df["Years"].max())),
    )
    sel_months = st.multiselect(
        "Months",
        options=MONTH_ORDER,
        default=MONTH_ORDER,
    )

    st.divider()
    st.markdown("### 🧪 Prediction Settings")
    station_dep = st.selectbox("🟢 Departure station", sorted(df["Departure station"].unique()))
    station_arr = st.selectbox("🔴 Arrival station",   sorted(df["Arrival station"].unique()))
    pred_day    = st.selectbox("Day of travel",  DAY_ORDER)
    pred_month  = st.selectbox("Month of travel", MONTH_ORDER)
    pred_year   = st.selectbox("Year",            sorted(df["Years"].unique(), reverse=True))

    st.divider()
    st.caption("Built by Kael")

#  FILTERED DATAFRAME
mask = (
    df["Service"].isin(sel_service) &
    df["Years"].between(sel_years[0], sel_years[1]) &
    df["Months"].isin(sel_months)
)
dff = df[mask].copy()

#  HEADER
st.markdown("""
<div style='padding: 24px 0 8px'>
  <h1 style='margin:0;font-size:2rem;font-weight:700;color:#f1f5f9'>
    🚄 SNCF Train Delay Prediction
  </h1>
  <p style='color:#64748b;margin:4px 0 0'>
    Real-time analytics & AI-powered delay prediction · French high-speed rail network
  </p>
</div>
""", unsafe_allow_html=True)

#  KPI ROW
total_trains    = int(dff["Number of scheduled trains"].sum())
total_cancelled = int(dff["Number of cancelled trains"].sum())
avg_delay       = dff["Average delay of all trains at arrival"].mean()
pct_on_time     = (dff["Statut"] == "On Time").mean() * 100
pct_major       = (dff["Statut"] == "Mayor delay").mean() * 100
cancel_rate     = (total_cancelled / total_trains * 100) if total_trains > 0 else 0

k1, k2, k3, k4, k5 = st.columns(5)
def kpi(col, val, label, delta_html=""):
    col.markdown(f"""
    <div class='kpi-card'>
      <div class='kpi-label'>{label}</div>
      <div class='kpi-value'>{val}</div>
      {delta_html}
    </div>""", unsafe_allow_html=True)

with k1: kpi(k1, f"{total_trains:,}", "Scheduled Trains")
with k2: kpi(k2, f"{avg_delay:.1f} min", "Avg Arrival Delay",
             f"<div class='kpi-delta delta-up'>⬆ across filtered period</div>")
with k3: kpi(k3, f"{total_cancelled:,}", "Cancelled",
             f"<div class='kpi-delta delta-up'>{cancel_rate:.1f}% cancellation rate</div>")
with k4: kpi(k4, f"{pct_on_time:.1f}%", "On Time",
             f"<div class='kpi-delta delta-down'>✓ target >10%</div>")
with k5: kpi(k5, f"{pct_major:.1f}%", "Major Delays",
             f"<div class='kpi-delta delta-up'>⚠ >15 min</div>")

st.markdown("<div style='margin-top:8px'></div>", unsafe_allow_html=True)

#  TABS
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview", "🗺️ Stations", "📅 Time Analysis",
    "⚙️ Delay Causes", "🤖 Prediction"
])

#  TAB 1 — OVERVIEW
with tab1:
    c1, c2 = st.columns(2)

    # Delay status distribution
    with c1:
        st.markdown("<div class='section-header'>Delay Status Distribution</div>", unsafe_allow_html=True)
        status_counts = dff["Statut"].value_counts().reset_index()
        status_counts.columns = ["Status", "Count"]
        color_map = {"On Time": "#4ade80", "Slight delay": AMBER, "Mayor delay": "#f87171"}
        fig = px.pie(
            status_counts, values="Count", names="Status",
            color="Status", color_discrete_map=color_map,
            hole=0.55,
        )
        fig.update_traces(
            textposition="outside", textinfo="percent+label",
            pull=[0.03, 0.03, 0.07],
            marker=dict(line=dict(color="#0f172a", width=2))
        )
        fig.update_layout(**PLOTLY_THEME, showlegend=False, height=300)
        st.plotly_chart(fig, use_container_width=True)

    # Delay by service
    with c2:
        st.markdown("<div class='section-header'>Avg Delay by Service</div>", unsafe_allow_html=True)
        svc = dff.groupby("Service")["Average delay of all trains at arrival"].mean().reset_index()
        svc.columns = ["Service", "Avg Delay"]
        fig2 = px.bar(
            svc, x="Service", y="Avg Delay",
            color="Avg Delay", color_continuous_scale="Blues",
            text="Avg Delay",
        )
        fig2.update_traces(texttemplate="%{text:.1f} min", textposition="outside")
        fig2.update_layout(**PLOTLY_THEME, coloraxis_showscale=False, height=300,
                           yaxis_title="Minutes", xaxis_title="")
        st.plotly_chart(fig2, use_container_width=True)

    # Yearly trend
    st.markdown("<div class='section-header'>Yearly Delay Trend</div>", unsafe_allow_html=True)
    yr = dff.groupby("Years").agg(
        avg_delay=("Average delay of all trains at arrival", "mean"),
        cancellations=("Number of cancelled trains", "sum"),
    ).reset_index()
    fig3 = make_subplots(specs=[[{"secondary_y": True}]])
    fig3.add_trace(go.Scatter(
        x=yr["Years"], y=yr["avg_delay"], name="Avg Delay (min)",
        line=dict(color=ACCENT, width=3), fill="tozeroy",
        fillcolor="rgba(56,189,248,.15)", mode="lines+markers",
        marker=dict(size=8, color=ACCENT)
    ), secondary_y=False)
    fig3.add_trace(go.Bar(
        x=yr["Years"], y=yr["cancellations"], name="Total Cancellations",
        marker_color="rgba(248,113,113,.6)", yaxis="y2"
    ), secondary_y=True)
    fig3.update_layout(
        **PLOTLY_THEME, height=340,
        legend=dict(orientation="h", y=1.1),
        yaxis=dict(title="Avg Delay (min)"),
        yaxis2=dict(title="Cancellations", overlaying="y", side="right"),
    )
    st.plotly_chart(fig3, use_container_width=True)

    # Distribution of arrival delay
    st.markdown("<div class='section-header'>Arrival Delay Distribution</div>", unsafe_allow_html=True)
    fig4 = px.histogram(
        dff, x="Average delay of all trains at arrival",
        nbins=60, color_discrete_sequence=[ACCENT],
        marginal="box",
    )
    fig4.update_layout(**PLOTLY_THEME, height=300,
                       xaxis_title="Avg Delay at Arrival (min)",
                       yaxis_title="Frequency")
    st.plotly_chart(fig4, use_container_width=True)

#  TAB 2 — STATIONS
with tab2:
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("<div class='section-header'>Top 15 Departure Stations — Avg Delay</div>", unsafe_allow_html=True)
        dep_stats = (
            dff.groupby("Departure station")["Average delay of all trains at arrival"]
            .mean().nlargest(15).sort_values()
            .reset_index()
        )
        dep_stats.columns = ["Station", "Avg Delay"]
        fig = px.bar(
            dep_stats, x="Avg Delay", y="Station", orientation="h",
            color="Avg Delay", color_continuous_scale="Reds",
            text="Avg Delay",
        )
        fig.update_traces(texttemplate="%{text:.1f} min", textposition="outside")
        fig.update_layout(**PLOTLY_THEME, coloraxis_showscale=False, height=450,
                          xaxis_title="Minutes", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("<div class='section-header'>Top 15 Arrival Stations — Avg Delay</div>", unsafe_allow_html=True)
        arr_stats = (
            dff.groupby("Arrival station")["Average delay of all trains at arrival"]
            .mean().nlargest(15).sort_values()
            .reset_index()
        )
        arr_stats.columns = ["Station", "Avg Delay"]
        fig2 = px.bar(
            arr_stats, x="Avg Delay", y="Station", orientation="h",
            color="Avg Delay", color_continuous_scale="Blues",
            text="Avg Delay",
        )
        fig2.update_traces(texttemplate="%{text:.1f} min", textposition="outside")
        fig2.update_layout(**PLOTLY_THEME, coloraxis_showscale=False, height=450,
                           xaxis_title="Minutes", yaxis_title="")
        st.plotly_chart(fig2, use_container_width=True)

    # Route heatmap: top routes
    st.markdown("<div class='section-header'>Most Delayed Routes (Top 20)</div>", unsafe_allow_html=True)
    routes = (
        dff.groupby(["Departure station", "Arrival station"])
        .agg(avg_delay=("Average delay of all trains at arrival", "mean"),
             count=("Date", "count"))
        .reset_index()
        .query("count >= 5")
        .nlargest(20, "avg_delay")
    )
    routes["Route"] = routes["Departure station"] + " → " + routes["Arrival station"]
    fig3 = px.bar(
        routes.sort_values("avg_delay"), x="avg_delay", y="Route",
        orientation="h", color="avg_delay",
        color_continuous_scale="RdYlGn_r",
        hover_data={"count": True, "avg_delay": ":.2f"},
    )
    fig3.update_layout(**PLOTLY_THEME, coloraxis_showscale=False, height=500,
                       xaxis_title="Avg Delay (min)", yaxis_title="")
    st.plotly_chart(fig3, use_container_width=True)

    # Station cancellations
    st.markdown("<div class='section-header'>Top Stations by Cancellations</div>", unsafe_allow_html=True)
    can_stats = (
        dff.groupby("Departure station")["Number of cancelled trains"]
        .sum().nlargest(12).reset_index()
    )
    can_stats.columns = ["Station", "Cancellations"]
    fig4 = px.treemap(
        can_stats, path=["Station"], values="Cancellations",
        color="Cancellations", color_continuous_scale="Reds",
    )
    fig4.update_layout(**PLOTLY_THEME, height=380)
    st.plotly_chart(fig4, use_container_width=True)

#  TAB 3 — TIME ANALYSIS
with tab3:
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("<div class='section-header'>Avg Delay by Day of Week</div>", unsafe_allow_html=True)
        day_delay = (
            dff.groupby("Days", observed=True)["Average delay of all trains at arrival"]
            .mean().reindex(DAY_ORDER).reset_index()
        )
        day_delay.columns = ["Day", "Avg Delay"]
        colors = [RED_COLOR if d == "Friday" else ACCENT for d in day_delay["Day"]]
        fig = go.Figure(go.Bar(
            x=day_delay["Day"], y=day_delay["Avg Delay"],
            marker_color=colors,
            text=day_delay["Avg Delay"].round(2),
            texttemplate="%{text:.1f}",
            textposition="outside",
        ))
        fig.update_layout(**PLOTLY_THEME, height=320,
                          yaxis_title="Minutes", xaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("<div class='section-header'>Avg Delay by Month</div>", unsafe_allow_html=True)
        month_delay = (
            dff.groupby("Months", observed=True)["Average delay of all trains at arrival"]
            .mean().reindex(MONTH_ORDER).reset_index()
        )
        month_delay.columns = ["Month", "Avg Delay"]
        fig2 = px.line(
            month_delay, x="Month", y="Avg Delay",
            markers=True, line_shape="spline",
            color_discrete_sequence=[ACCENT],
        )
        fig2.update_traces(fill="tozeroy", fillcolor="rgba(56,189,248,.1)",
                           line=dict(width=3), marker=dict(size=9))
        fig2.update_layout(**PLOTLY_THEME, height=320,
                           yaxis_title="Minutes", xaxis_title="")
        st.plotly_chart(fig2, use_container_width=True)

    # Monthly cancellations
    st.markdown("<div class='section-header'>Monthly Cancellations Over Time</div>", unsafe_allow_html=True)
    monthly_cancel = (
        dff.groupby(["Years", "Months"], observed=True)["Number of cancelled trains"]
        .sum().reset_index()
    )
    monthly_cancel["Period"] = monthly_cancel["Years"].astype(str) + "-" + monthly_cancel["Months"].astype(str)
    fig3 = px.area(
        monthly_cancel, x="Period", y="Number of cancelled trains",
        color="Years", line_group="Years",
        color_discrete_sequence=px.colors.sequential.Blues_r,
    )
    fig3.update_layout(**PLOTLY_THEME, height=330,
                       xaxis_title="", yaxis_title="Cancellations",
                       xaxis=dict(tickangle=45))
    st.plotly_chart(fig3, use_container_width=True)

    # Day × Month heatmap
    st.markdown("<div class='section-header'>Delay Heatmap: Day × Month</div>", unsafe_allow_html=True)
    heat = (
        dff.groupby(["Days", "Months"], observed=True)["Average delay of all trains at arrival"]
        .mean().unstack("Months").reindex(index=DAY_ORDER)
    )
    heat = heat.reindex(columns=MONTH_ORDER)
    fig4 = px.imshow(
        heat, color_continuous_scale="RdYlGn_r",
        aspect="auto", text_auto=".1f",
        labels=dict(color="Avg Delay (min)")
    )
    fig4.update_layout(**PLOTLY_THEME, height=350, coloraxis_colorbar=dict(title="min"))
    st.plotly_chart(fig4, use_container_width=True)

#  TAB 4 — DELAY CAUSES
with tab4:
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("<div class='section-header'>Average Cause Breakdown (All Data)</div>", unsafe_allow_html=True)
        causes_mean = {k: dff[v].mean() for k, v in CAUSE_COLS.items()}
        causes_df   = pd.DataFrame(list(causes_mean.items()), columns=["Cause", "Pct"])
        fig = px.bar(
            causes_df.sort_values("Pct"), x="Pct", y="Cause", orientation="h",
            color="Pct", color_continuous_scale="Blues",
            text="Pct",
        )
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig.update_layout(**PLOTLY_THEME, coloraxis_showscale=False, height=380,
                          xaxis_title="Average %", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("<div class='section-header'>Cause Breakdown — Radar</div>", unsafe_allow_html=True)
        cats  = list(causes_mean.keys())
        vals  = list(causes_mean.values())
        fig2 = go.Figure(go.Scatterpolar(
            r=vals + [vals[0]],
            theta=cats + [cats[0]],
            fill="toself",
            fillcolor="rgba(56,189,248,.2)",
            line=dict(color=ACCENT, width=2.5),
        ))
        fig2.update_layout(**PLOTLY_THEME, height=380,
                           polar=dict(
                               bgcolor="rgba(0,0,0,0)",
                               radialaxis=dict(gridcolor="#334155", tickcolor="#64748b"),
                               angularaxis=dict(gridcolor="#334155"),
                           ))
        st.plotly_chart(fig2, use_container_width=True)

    # Cause evolution over years
    st.markdown("<div class='section-header'>Delay Causes Evolution by Year</div>", unsafe_allow_html=True)
    cause_yr = dff.groupby("Years")[[v for v in CAUSE_COLS.values()]].mean().reset_index()
    cause_yr_melt = cause_yr.melt(id_vars="Years", var_name="Cause", value_name="Pct")
    # shorten labels
    label_map = {v: k for k, v in CAUSE_COLS.items()}
    cause_yr_melt["Cause"] = cause_yr_melt["Cause"].map(label_map)
    fig3 = px.line(
        cause_yr_melt, x="Years", y="Pct", color="Cause",
        markers=True, line_shape="spline",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig3.update_traces(line_width=2.5, marker_size=7)
    fig3.update_layout(**PLOTLY_THEME, height=380,
                       yaxis_title="Average %", xaxis_title="Year",
                       legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig3, use_container_width=True)

    # Correlation: cause → delay
    st.markdown("<div class='section-header'>Correlation: Causes vs Arrival Delay</div>", unsafe_allow_html=True)
    corr_cols = list(CAUSE_COLS.values()) + ["Average delay of all trains at arrival"]
    corr = dff[corr_cols].corr()
    corr_target = corr["Average delay of all trains at arrival"].drop(
        "Average delay of all trains at arrival"
    ).reset_index()
    corr_target.columns = ["Feature", "Correlation"]
    corr_target["Feature"] = corr_target["Feature"].map(label_map)
    corr_target = corr_target.sort_values("Correlation")
    colors = [RED_COLOR if v > 0 else "#4ade80" for v in corr_target["Correlation"]]
    fig4 = go.Figure(go.Bar(
        x=corr_target["Correlation"], y=corr_target["Feature"],
        orientation="h", marker_color=colors,
        text=corr_target["Correlation"].round(3),
        textposition="outside",
    ))
    fig4.update_layout(**PLOTLY_THEME, height=320,
                       xaxis_title="Pearson r", yaxis_title="",
                       xaxis=dict(range=[-0.5, 0.5]))
    st.plotly_chart(fig4, use_container_width=True)

#  TAB 5 — PREDICTION
with tab5:
    st.markdown("""
    <div style='background:linear-gradient(135deg,#1e3a5f,#1e293b);border:1px solid #334155;
         border-radius:16px;padding:20px 24px;margin-bottom:24px'>
      <h3 style='color:#38bdf8;margin:0 0 6px'> Delay Predictor</h3>
      <p style='color:#94a3b8;margin:0;font-size:.9rem'>
        Enter operational parameters below and click <b style='color:#f1f5f9'>Predict Delay</b> 
        to get a real-time estimate powered by Random Forest.
      </p>
    </div>
    """, unsafe_allow_html=True)

    if model is None:
        st.warning("⚠️ model.joblib not found — run `tardis_model.ipynb` first to train & save the model.")
    else:
        # ── Route info (from sidebar)
        col_info = st.columns(3)
        col_info[0].info(f"🟢 **Departure:** {station_dep}")
        col_info[1].info(f"🔴 **Arrival:** {station_arr}")
        col_info[2].info(f"📅 **{pred_day}, {pred_month} {pred_year}**")

        # ── Operational sliders
        st.markdown("<div class='section-header'>Operational Parameters</div>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)

        with col1:
            n_scheduled   = st.slider("Scheduled trains",
                int(df["Number of scheduled trains"].min()),
                int(df["Number of scheduled trains"].max()),
                int(df["Number of scheduled trains"].median()))
            n_cancelled   = st.slider("Cancelled trains",
                int(df["Number of cancelled trains"].min()),
                int(df["Number of cancelled trains"].max()),
                int(df["Number of cancelled trains"].median()))
            n_delayed_dep = st.slider("Trains delayed at departure",
                int(df["Number of trains delayed at departure"].min()),
                int(df["Number of trains delayed at departure"].max()),
                int(df["Number of trains delayed at departure"].median()))
            avg_delay_dep = st.slider("Avg delay late trains at departure (min)",
                float(df["Average delay of late trains at departure"].min()),
                float(df["Average delay of late trains at departure"].max()),
                float(df["Average delay of late trains at departure"].median()))

        with col2:
            n_15 = st.slider("Trains delayed > 15 min",
                int(df["Number of trains delayed > 15min"].min()),
                int(df["Number of trains delayed > 15min"].max()),
                int(df["Number of trains delayed > 15min"].median()))
            n_30 = st.slider("Trains delayed > 30 min",
                int(df["Number of trains delayed > 30min"].min()),
                int(df["Number of trains delayed > 30min"].max()),
                int(df["Number of trains delayed > 30min"].median()))
            n_60 = st.slider("Trains delayed > 60 min",
                int(df["Number of trains delayed > 60min"].min()),
                int(df["Number of trains delayed > 60min"].max()),
                int(df["Number of trains delayed > 60min"].median()))

        st.markdown("<div class='section-header'>Delay Cause Breakdown (%)</div>", unsafe_allow_html=True)
        p1, p2, p3, p4 = st.columns(4)
        with p1: pct_ext  = st.number_input("External causes", 0.0, 100.0, float(df["Pct delay due to external causes"].mean()), step=0.5)
        with p2: pct_inf  = st.number_input("Infrastructure",  0.0, 100.0, float(df["Pct delay due to infrastructure"].mean()), step=0.5)
        with p3: pct_traf = st.number_input("Traffic mgmt",    0.0, 100.0, float(df["Pct delay due to traffic management"].mean()), step=0.5)
        with p4: pct_roll = st.number_input("Rolling stock",   0.0, 100.0, float(df["Pct delay due to rolling stock"].mean()), step=0.5)

        # ── Predict button
        st.markdown("<div style='margin-top:16px'></div>", unsafe_allow_html=True)
        if st.button("🚄 Predict Delay", type="primary", use_container_width=True):
            try:
                dep_enc = station_encoder.transform([station_dep.upper()])[0]
                arr_enc = station_encoder.transform([station_arr.upper()])[0]
            except ValueError as e:
                st.error(f"Encoding error: {e}")
                st.stop()

            input_df = pd.DataFrame([{
                "Departure station": dep_enc,
                "Arrival station":   arr_enc,
                "Number of scheduled trains":                n_scheduled,
                "Number of cancelled trains":                n_cancelled,
                "Number of trains delayed at departure":     n_delayed_dep,
                "Average delay of late trains at departure": avg_delay_dep,
                "Number of trains delayed > 15min":          n_15,
                "Number of trains delayed > 30min":          n_30,
                "Number of trains delayed > 60min":          n_60,
                "Pct delay due to external causes":          pct_ext,
                "Pct delay due to infrastructure":           pct_inf,
                "Pct delay due to traffic management":       pct_traf,
                "Pct delay due to rolling stock":            pct_roll,
                "month":       MONTH_MAP[pred_month],
                "year":        pred_year,
                "day_of_week": DAY_MAP[pred_day],
            }])[FEATURES]

            pred = model.predict(input_df)[0]

            # Badge
            if pred < 5:
                badge_class, icon, label = "badge-green",  "✅", "ON TIME"
            elif pred < 15:
                badge_class, icon, label = "badge-orange", "⚠️", "SLIGHT DELAY"
            else:
                badge_class, icon, label = "badge-red",    "🚨", "MAJOR DELAY"

            r1, r2, r3 = st.columns([1, 2, 1])
            with r2:
                st.markdown(f"""
                <div class='pred-badge {badge_class}'>
                  {icon} {label}<br>
                  <span style='font-size:2.8rem'>{pred:.1f} min</span><br>
                  <span style='font-size:0.9rem;opacity:.7'>estimated arrival delay</span>
                </div>
                """, unsafe_allow_html=True)

            # Gauge chart
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=pred,
                title=dict(text="Estimated Delay (min)", font=dict(color="#94a3b8", size=14)),
                number=dict(suffix=" min", font=dict(color="#f1f5f9", size=32)),
                delta=dict(reference=dff["Average delay of all trains at arrival"].mean(),
                           valueformat=".1f",
                           increasing=dict(color=RED_COLOR),
                           decreasing=dict(color="#4ade80")),
                gauge=dict(
                    axis=dict(range=[0, 30], tickcolor="#64748b"),
                    bar=dict(color=ACCENT),
                    bgcolor="rgba(0,0,0,0)",
                    bordercolor="#334155",
                    steps=[
                        dict(range=[0, 5],   color="rgba(74,222,128,.2)"),
                        dict(range=[5, 15],  color="rgba(251,191,36,.2)"),
                        dict(range=[15, 30], color="rgba(248,113,113,.2)"),
                    ],
                    threshold=dict(
                        line=dict(color=RED_COLOR, width=3),
                        thickness=0.8,
                        value=dff["Average delay of all trains at arrival"].mean()
                    ),
                ),
            ))
            fig_gauge.update_layout(**PLOTLY_THEME, height=320)
            st.plotly_chart(fig_gauge, use_container_width=True)

            # Historical comparison for this route
            route_hist = df[
                (df["Departure station"] == station_dep) &
                (df["Arrival station"]   == station_arr)
            ]
            if len(route_hist) > 0:
                st.markdown("<div class='section-header'>Historical Data for This Route</div>", unsafe_allow_html=True)
                rh_month = (
                    route_hist.groupby("Months", observed=True)["Average delay of all trains at arrival"]
                    .mean().reindex(MONTH_ORDER).reset_index()
                )
                fig_rh = px.bar(
                    rh_month, x="Months", y="Average delay of all trains at arrival",
                    color="Average delay of all trains at arrival",
                    color_continuous_scale="RdYlGn_r",
                    text="Average delay of all trains at arrival",
                )
                fig_rh.add_hline(y=pred, line_color=ACCENT, line_dash="dash",
                                 annotation_text=f"Your prediction: {pred:.1f} min",
                                 annotation_font_color=ACCENT)
                fig_rh.update_traces(texttemplate="%{text:.1f}", textposition="outside")
                fig_rh.update_layout(**PLOTLY_THEME, height=300,
                                     coloraxis_showscale=False,
                                     yaxis_title="Avg Delay (min)", xaxis_title="")
                st.plotly_chart(fig_rh, use_container_width=True)
            else:
                st.info("No historical data available for this specific route combination.")

#  FOOTER — Data Preview
with st.expander("📁 Raw Dataset Preview (first 50 rows)", expanded=False):
    st.dataframe(
        dff.head(50).style.format(precision=2),
        use_container_width=True, height=350
    )

st.markdown("""
<div style='text-align:center;color:#334155;padding:32px 0 16px;font-size:.78rem'>
  TARDIS · SNCF Delay Prediction Platform 
</div>
""", unsafe_allow_html=True)
