# Influencer Network Lifetime Value & Churn Mitigation Matrix

An actuarial-grade analytics dashboard that exposes the true lifetime value and churn velocity of bettors acquired through influencer marketing campaigns — built for sports betting agency leadership.

---

## The Problem

Sports betting agencies pay influencer retainers based on conversion volume. But conversions are a vanity metric. What matters is how long an acquired bettor stays active and how much revenue they generate over their lifetime.

This dashboard answers the question traditional marketing analytics never asks:

> *What is the probability this bettor is still active on Day 30? Day 60? Day 90?*

---

## What It Does

- Synthesises a cohort of **2,500 anonymised acquired bettors** across 5 influencer archetypes
- Fits a **Kaplan-Meier Survival Model** per creator node to calculate bettor retention probability over a 90-day horizon
- Computes **Projected CLV in ZMW** using variance-weighted monthly deposit behaviour
- Calculates **ROI per influencer node** after deducting CPA acquisition costs
- Renders an interactive dark-mode dashboard with dynamic filtering by creator node

---

## Key Findings (Synthetic Data)

| Creator Node | ROI % | Median Churn Day | Avg CLV (ZMW) |
|---|---|---|---|
| Football Tipping Expert | 204.3% | Day 62 | K 1,157 |
| Niche Sports Analyst | 184.1% | Day 79 | K 1,078 |
| Flashy Lifestyle Creator | 65.1% | Day 40 | K 594 |
| GenZ Gamer | 36.2% | Day 39 | K 549 |
| Viral Comedy Creator | 17.7% | Day 24 | K 335 |

Same retainer. **11x difference in value delivered.**

---

## Dashboard Sections

1. **KPI Cards** — Total conversions, Average CLV, 90-Day churn velocity
2. **Kaplan-Meier Survival Curves** — Comparative decay curves across all acquisition nodes
3. **Total Revenue & ROI Chart** — Gross CLV vs Net Revenue after CPA per node
4. **Player Acquisition Distribution** — Conversion volume and network share per creator
5. **CLV Distribution Histogram** — Spread of individual bettor CLV values per node
6. **Deposit vs Retention Scatter** — Player-level deposit behaviour against retention duration
7. **Capital Generation Summary Table** — Node-level actuarial summary
8. **Raw Transaction Logs** — Underlying deposit ledger, filterable and paginated

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core data engine |
| Preswald | Dashboard framework |
| lifelines | Kaplan-Meier survival analysis |
| Plotly | Interactive charts |
| Pandas / NumPy | Data manipulation |

---

## How to Run

```bash
# Install dependencies
pip install preswald lifelines plotly pandas numpy

# Clone the repo
git clone https://github.com/MahlonMeleki/influencer-clv-dashboard.git
cd influencer-clv-dashboard

# Run the dashboard
preswald run
```

Then open your browser at `http://localhost:8501`

---

## Methodology

**Survival Analysis**
Bettor churn is modelled as a survival problem. Each creator node's acquisition cohort is fitted with a Kaplan-Meier estimator using the `lifelines` library. The survival function S(t) gives the probability a bettor remains active at day t.

**CLV Computation**
```
Projected CLV = Tenure (months) × Variance-Weighted Monthly Deposit × Revenue Share Rate (30%)
```

Variance-weighting adjusts for deposit volatility across months, producing a more conservative and realistic CLV estimate than simple averages.

**Synthetic Data**
All player data is procedurally generated using archetype-specific exponential hazard rates and normal deposit distributions. No real player data is used.

---

## Author

**Mahlon Meleki**  
Actuarial Science Student | University of Zambia  
Quantitative Finance | Risk Analytics | Data Science  

[LinkedIn](https://www.linkedin.com/in/mahlonmeleki) · [GitHub](https://github.com/MahlonMeleki)
