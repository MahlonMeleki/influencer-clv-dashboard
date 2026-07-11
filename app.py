import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import preswald as pw
from lifelines import KaplanMeierFitter

if not hasattr(pw, "metric"):
    pw.metric = pw.big_number
if not hasattr(pw, "dataframe"):
    pw.dataframe = pw.table

RNG = np.random.default_rng(seed=42)

CREATORS = [
    "Niche Sports Analyst",
    "Football Tipping Expert",
    "Flashy Lifestyle Creator",
    "Viral Comedy Creator",
    "GenZ Gamer",
]

ARCHETYPE_PARAMS = {
    "Niche Sports Analyst":     {"halflife": 72, "dep_mean": 1850, "dep_std": 420,  "weight": 0.18},
    "Football Tipping Expert":  {"halflife": 65, "dep_mean": 2100, "dep_std": 510,  "weight": 0.26},
    "Flashy Lifestyle Creator": {"halflife": 38, "dep_mean": 1400, "dep_std": 680,  "weight": 0.22},
    "Viral Comedy Creator":     {"halflife": 28, "dep_mean": 950,  "dep_std": 350,  "weight": 0.20},
    "GenZ Gamer":               {"halflife": 45, "dep_mean": 1150, "dep_std": 290,  "weight": 0.14},
}

N_PLAYERS = 2500
OBSERVATION_WINDOW = 90
REVENUE_SHARE_RATE = 0.30
ACQUISITION_COST_ZMW = 380

PALETTE = {
    "Niche Sports Analyst":     "#00D4FF",
    "Football Tipping Expert":  "#FFD700",
    "Flashy Lifestyle Creator": "#FF6B6B",
    "Viral Comedy Creator":     "#7BFF6E",
    "GenZ Gamer":               "#C77DFF",
}

DARK_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="#0D1117",
    plot_bgcolor="#0D1117",
    font=dict(family="Inter, sans-serif", color="#E6EDF3"),
    legend=dict(bgcolor="#161B22", bordercolor="#30363D", borderwidth=1),
)

ARCHETYPE_VAR_FLOOR = 2500.0


def variance_weighted_monthly_deposit(deposits: np.ndarray) -> float:
    deposits = np.asarray(deposits, dtype=float)
    if deposits.size == 0:
        return 0.0
    if deposits.size == 1:
        return float(deposits[0])
    spread = (deposits - deposits.mean()) ** 2
    weights = 1.0 / (spread + ARCHETYPE_VAR_FLOOR)
    return float(np.average(deposits, weights=weights))


# ---------------------------------------------------------------------------
# DATA ENGINE
# ---------------------------------------------------------------------------
creator_assignments = RNG.choice(
    CREATORS,
    size=N_PLAYERS,
    p=[ARCHETYPE_PARAMS[c]["weight"] for c in CREATORS],
)

registration_offsets = RNG.integers(0, OBSERVATION_WINDOW, size=N_PLAYERS)
registration_dates = pd.to_datetime("2024-01-01") + pd.to_timedelta(registration_offsets, unit="D")

players = []
transaction_rows = []
txn_counter = 1

for idx, creator in enumerate(creator_assignments):
    params = ARCHETYPE_PARAMS[creator]
    hazard_lambda = np.log(2) / params["halflife"]
    raw_duration = RNG.exponential(scale=1 / hazard_lambda)
    duration_days = float(min(raw_duration, OBSERVATION_WINDOW))
    churned = int(raw_duration <= OBSERVATION_WINDOW)
    reg_date = registration_dates[idx]
    last_active = reg_date + pd.Timedelta(days=int(duration_days))
    tenure_months = duration_days / 30.44

    active_months = max(1, int(np.ceil(tenure_months)))
    monthly_deposits = np.clip(
        RNG.normal(loc=params["dep_mean"], scale=params["dep_std"], size=active_months),
        150, None,
    )
    vw_monthly_deposit = variance_weighted_monthly_deposit(monthly_deposits)
    projected_clv = tenure_months * vw_monthly_deposit * REVENUE_SHARE_RATE

    player_id = f"PLY-{idx + 1:05d}"
    players.append({
        "Player_ID": player_id,
        "Registration_Date": reg_date.strftime("%Y-%m-%d"),
        "Last_Active_Date": last_active.strftime("%Y-%m-%d"),
        "Creator_Node": creator,
        "Duration_Days": round(duration_days, 2),
        "Churned": churned,
        "Tenure_Months": round(tenure_months, 3),
        "Variance_Weighted_Monthly_Deposit_ZMW": round(vw_monthly_deposit, 2),
        "Projected_CLV_ZMW": round(projected_clv, 2),
    })

    for month_idx, amount in enumerate(monthly_deposits):
        txn_date = reg_date + pd.DateOffset(months=month_idx)
        if txn_date > last_active:
            break
        transaction_rows.append({
            "Transaction_ID": f"TXN-{txn_counter:06d}",
            "Player_ID": player_id,
            "Creator_Node": creator,
            "Transaction_Date": txn_date.strftime("%Y-%m-%d"),
            "Transaction_Type": "Deposit",
            "Amount_ZMW": round(float(amount), 2),
        })
        txn_counter += 1

df = pd.DataFrame(players)
transactions = pd.DataFrame(transaction_rows)

# ---------------------------------------------------------------------------
# KAPLAN-MEIER ENGINE
# ---------------------------------------------------------------------------
km_resolution = np.linspace(0, OBSERVATION_WINDOW, 181)
km_curves = {}
median_survival_days = {}

for creator in CREATORS:
    mask = df["Creator_Node"] == creator
    kmf = KaplanMeierFitter()
    kmf.fit(
        durations=df.loc[mask, "Duration_Days"],
        event_observed=df.loc[mask, "Churned"],
        label=creator,
    )
    km_curves[creator] = kmf.predict(km_resolution).values
    median_survival_days[creator] = round(float(kmf.median_survival_time_), 1)

km_long = pd.DataFrame(km_curves, index=km_resolution).reset_index(names="Day").melt(
    id_vars="Day",
    var_name="Creator_Node",
    value_name="Survival_Probability",
)

# ---------------------------------------------------------------------------
# NODE STATS
# ---------------------------------------------------------------------------
node_stats = (
    df.groupby("Creator_Node", as_index=False)
    .agg(
        Conversions=("Player_ID", "count"),
        Avg_CLV_ZMW=("Projected_CLV_ZMW", "mean"),
        Total_CLV_ZMW=("Projected_CLV_ZMW", "sum"),
        Avg_Monthly_Deposit=("Variance_Weighted_Monthly_Deposit_ZMW", "mean"),
        Churn_Rate=("Churned", "mean"),
    )
)
node_stats["Total_Acquisition_Cost_ZMW"] = node_stats["Conversions"] * ACQUISITION_COST_ZMW
node_stats["Net_Revenue_ZMW"] = node_stats["Total_CLV_ZMW"] - node_stats["Total_Acquisition_Cost_ZMW"]
node_stats["ROI_Pct"] = (
    node_stats["Net_Revenue_ZMW"] / node_stats["Total_Acquisition_Cost_ZMW"] * 100
).round(1)
node_stats["Survival_90D"] = [round(km_curves[c][-1] * 100, 1) for c in node_stats["Creator_Node"]]
node_stats["Median_Survival_Days"] = [median_survival_days[c] for c in node_stats["Creator_Node"]]
node_stats["CLV_Per_Day"] = (node_stats["Avg_Monthly_Deposit"] * REVENUE_SHARE_RATE) / 30.44
node_stats["Breakeven_Day"] = (ACQUISITION_COST_ZMW / node_stats["CLV_Per_Day"]).round(1)

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
pw.sidebar(defaultopen=True, name="Influencer CLV Matrix")

pw.text("""
# Influencer Network Lifetime Value & Churn Mitigation Matrix
**Parametric Risk Intelligence** | 2,500-Acquired Bettor Cohort | ZMW Denomination
""")

pw.separator()

selected_node = pw.selectbox(
    label="Isolate Creator Acquisition Node",
    options=["All Nodes"] + CREATORS,
    default="All Nodes",
)

if selected_node != "All Nodes":
    df_view = df[df["Creator_Node"] == selected_node].copy()
    txn_view = transactions[transactions["Creator_Node"] == selected_node].copy()
    km_view = km_long[km_long["Creator_Node"] == selected_node].copy()
    node_view = node_stats[node_stats["Creator_Node"] == selected_node].copy()
    survival_90d = km_curves[selected_node][-1] * 100
else:
    df_view = df.copy()
    txn_view = transactions.copy()
    km_view = km_long.copy()
    node_view = node_stats.copy()
    survival_90d = np.mean([km_curves[c][-1] for c in CREATORS]) * 100

total_conversions = len(df_view)
avg_clv = df_view["Projected_CLV_ZMW"].mean()
churn_velocity = df_view["Churned"].mean() * 100

# KPI CARDS
pw.metric(
    value=f"{total_conversions:,}",
    label="Total Conversions",
    delta=selected_node,
    size=0.33,
)
pw.metric(
    value=f"K {avg_clv:,.0f}",
    label="Average Projected User CLV",
    delta="Variance-weighted deposit model",
    delta_color="green",
    size=0.33,
)
pw.metric(
    value=f"{churn_velocity:.1f}%",
    label="90-Day User Churn Velocity",
    delta=f"{survival_90d:.1f}% survival at day 90",
    delta_color="red" if churn_velocity > 55 else "normal",
    size=0.33,
)

pw.separator()

# ---------------------------------------------------------------------------
# 1. KAPLAN-MEIER SURVIVAL CURVES
# ---------------------------------------------------------------------------
pw.text("### Kaplan-Meier Survival Curves by Acquisition Vector")

fig_km = px.line(
    km_view,
    x="Day",
    y="Survival_Probability",
    color="Creator_Node",
    color_discrete_map=PALETTE,
    markers=False,
    labels={
        "Day": "Days Since Registration",
        "Survival_Probability": "Probability Player Remains Active",
        "Creator_Node": "Creator Node",
    },
)
fig_km.update_layout(
    **DARK_LAYOUT,
    title_font=dict(size=16, color="#E6EDF3"),
    xaxis=dict(gridcolor="#21262D", range=[0, OBSERVATION_WINDOW]),
    yaxis=dict(gridcolor="#21262D", tickformat=".0%"),
    hovermode="x unified",
    height=400,
)
fig_km.add_hline(
    y=0.5,
    line_dash="dot",
    line_color="#58A6FF",
    annotation_text="Median survival threshold",
    annotation_font_color="#58A6FF",
)
pw.plotly(fig_km, component_id="km_chart")

pw.separator()

# ---------------------------------------------------------------------------
# 2. TOTAL REVENUE / ROI BY INFLUENCER
# ---------------------------------------------------------------------------
pw.text("### Total Revenue & ROI by Influencer Node")

roi_bar_data = node_view.sort_values("ROI_Pct", ascending=True).copy()

fig_revenue = go.Figure()
fig_revenue.add_trace(go.Bar(
    name="Gross CLV (ZMW)",
    y=roi_bar_data["Creator_Node"].tolist(),
    x=roi_bar_data["Total_CLV_ZMW"].tolist(),
    orientation="h",
    marker_color="#00D4FF",
    text=[f"K {v/1e3:.0f}K" for v in roi_bar_data["Total_CLV_ZMW"]],
    textposition="outside",
    textfont=dict(color="#E6EDF3", size=10),
))
fig_revenue.add_trace(go.Bar(
    name="Net Revenue After CPA (ZMW)",
    y=roi_bar_data["Creator_Node"].tolist(),
    x=roi_bar_data["Net_Revenue_ZMW"].tolist(),
    orientation="h",
    marker_color="#FFD700",
    text=[f"K {v/1e3:.0f}K" for v in roi_bar_data["Net_Revenue_ZMW"]],
    textposition="outside",
    textfont=dict(color="#E6EDF3", size=10),
))
fig_revenue.update_layout(
    template="plotly_dark",
    paper_bgcolor="#0D1117",
    plot_bgcolor="#0D1117",
    font=dict(family="Inter, sans-serif", color="#E6EDF3"),
    legend=dict(bgcolor="#161B22", bordercolor="#30363D", borderwidth=1),
    barmode="group",
    height=380,
    margin=dict(l=160, r=120, t=30, b=40),
    xaxis=dict(title="Capital (ZMW)", gridcolor="#21262D", tickformat=",.0f"),
    yaxis=dict(gridcolor="#21262D"),
    hovermode="y unified",
)
pw.plotly(fig_revenue, component_id="revenue_chart")

pw.separator()

# ---------------------------------------------------------------------------
# 3. PLAYER ACQUISITION DISTRIBUTION
# ---------------------------------------------------------------------------
pw.text("### Player Acquisition Distribution by Creator Node")

acq_data = node_view.sort_values("Conversions", ascending=False).copy()
colors_acq = [PALETTE[c] for c in acq_data["Creator_Node"]]

fig_acq = go.Figure()
fig_acq.add_trace(go.Bar(
    x=acq_data["Creator_Node"].tolist(),
    y=acq_data["Conversions"].tolist(),
    marker_color=colors_acq,
    text=acq_data["Conversions"].tolist(),
    textposition="outside",
    textfont=dict(color="#E6EDF3", size=11),
    showlegend=False,
))
fig_acq.add_trace(go.Scatter(
    x=acq_data["Creator_Node"].tolist(),
    y=(acq_data["Conversions"] / acq_data["Conversions"].sum() * 100).tolist(),
    mode="markers+text",
    name="Share %",
    yaxis="y2",
    marker=dict(color="#FF6B6B", size=12, symbol="circle"),
    text=[(f"{v:.1f}%") for v in (acq_data["Conversions"] / acq_data["Conversions"].sum() * 100)],
    textposition="top center",
    textfont=dict(color="#FF6B6B", size=10),
))
fig_acq.update_layout(
    template="plotly_dark",
    paper_bgcolor="#0D1117",
    plot_bgcolor="#0D1117",
    font=dict(family="Inter, sans-serif", color="#E6EDF3"),
    legend=dict(bgcolor="#161B22", bordercolor="#30363D", borderwidth=1),
    height=380,
    margin=dict(t=30, b=60),
    xaxis=dict(gridcolor="#21262D"),
    yaxis=dict(title="Number of Bettors", gridcolor="#21262D"),
    yaxis2=dict(
        title="Network Share %",
        overlaying="y",
        side="right",
        showgrid=False,
        zeroline=False,
        range=[0, (acq_data["Conversions"] / acq_data["Conversions"].sum() * 100).max() * 1.5],
    ),
    hovermode="x unified",
)
pw.plotly(fig_acq, component_id="acq_chart")

pw.separator()

# ---------------------------------------------------------------------------
# 4. CLV DISTRIBUTION HISTOGRAM
# ---------------------------------------------------------------------------
pw.text("### CLV Distribution by Creator Node")

fig_clv_hist = px.histogram(
    df_view,
    x="Projected_CLV_ZMW",
    color="Creator_Node",
    color_discrete_map=PALETTE,
    nbins=40,
    barmode="overlay",
    opacity=0.75,
    labels={
        "Projected_CLV_ZMW": "Projected CLV (ZMW)",
        "Creator_Node": "Creator Node",
        "count": "Number of Players",
    },
)
fig_clv_hist.update_layout(
    **DARK_LAYOUT,
    height=400,
    margin=dict(t=30, b=60),
    xaxis=dict(gridcolor="#21262D", title="Projected CLV (ZMW)"),
    yaxis=dict(gridcolor="#21262D", title="Number of Players"),
    hovermode="x unified",
)
fig_clv_hist.add_vline(
    x=avg_clv,
    line_dash="dot",
    line_color="#FF6B6B",
    annotation_text=f"  Network Avg: K {avg_clv:,.0f}",
    annotation_font_color="#FF6B6B",
)
pw.plotly(fig_clv_hist, component_id="clv_hist")

pw.separator()

# ---------------------------------------------------------------------------
# 5. DEPOSIT VS RETENTION SCATTER
# ---------------------------------------------------------------------------
pw.text("### Deposit Behaviour vs Retention — Player Level")

fig_scatter = px.scatter(
    df_view,
    x="Variance_Weighted_Monthly_Deposit_ZMW",
    y="Duration_Days",
    color="Creator_Node",
    color_discrete_map=PALETTE,
    opacity=0.55,
    size_max=8,
    labels={
        "Variance_Weighted_Monthly_Deposit_ZMW": "Monthly Deposit (ZMW)",
        "Duration_Days": "Retention Duration (Days)",
        "Creator_Node": "Creator Node",
    },
    hover_data=["Player_ID", "Projected_CLV_ZMW"],
)
fig_scatter.update_traces(marker=dict(size=5))
fig_scatter.update_layout(
    **DARK_LAYOUT,
    height=420,
    margin=dict(t=30, b=60),
    xaxis=dict(gridcolor="#21262D", title="Monthly Deposit (ZMW)"),
    yaxis=dict(gridcolor="#21262D", title="Retention Duration (Days)"),
    hovermode="closest",
)
fig_scatter.add_hline(
    y=30,
    line_dash="dot",
    line_color="#58A6FF",
    annotation_text="  30-Day retention floor",
    annotation_font_color="#58A6FF",
)
pw.plotly(fig_scatter, component_id="scatter_chart")

pw.separator()

# ---------------------------------------------------------------------------
# CAPITAL GENERATION SUMMARY TABLE
# ---------------------------------------------------------------------------
pw.text("### Capital Generation & ROI Summary")

roi_display = node_view[[
    "Creator_Node", "Conversions", "Total_CLV_ZMW",
    "Net_Revenue_ZMW", "ROI_Pct", "Survival_90D", "Median_Survival_Days"
]].copy().sort_values("ROI_Pct", ascending=False)

roi_display["Total_CLV_ZMW"] = roi_display["Total_CLV_ZMW"].map(lambda v: f"K {v:,.0f}")
roi_display["Net_Revenue_ZMW"] = roi_display["Net_Revenue_ZMW"].map(lambda v: f"K {v:,.0f}")
roi_display["ROI_Pct"] = roi_display["ROI_Pct"].map(lambda v: f"{v:.1f}%")
roi_display["Survival_90D"] = roi_display["Survival_90D"].map(lambda v: f"{v:.1f}%")
roi_display["Median_Survival_Days"] = roi_display["Median_Survival_Days"].map(lambda v: f"Day {v:.0f}")
roi_display.columns = [
    "Creator Node", "Conversions", "Gross CLV (ZMW)",
    "Net Revenue (ZMW)", "ROI %", "90D Survival %", "Median Churn Day"
]
pw.table(roi_display, component_id="roi_table")

pw.separator()

# ---------------------------------------------------------------------------
# RAW TRANSACTION LOGS
# ---------------------------------------------------------------------------
pw.text("### Raw Transaction Logs")

txn_display = txn_view.sort_values(
    ["Transaction_Date", "Player_ID"], ascending=[False, True]
).copy()
txn_display["Amount_ZMW"] = txn_display["Amount_ZMW"].map(lambda v: f"K {v:,.2f}")

pw.dataframe(
    txn_display,
    title="Underlying Deposit Transaction Ledger",
    limit=250,
)

pw.separator()
pw.text("""
---
*CLV = Tenure (months) × Variance-Weighted Monthly Deposit × 30% Revenue Share.  
Survival analysis: Kaplan-Meier estimator. Observation window: 90 days.*
""")