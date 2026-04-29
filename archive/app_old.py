"""
app.py — Demand Forecasting Ensemble · Gradio Interface
--------------------------------------------------------
Run AFTER the notebook has executed and saved model_artifacts.pkl:

    python app.py
"""

import pickle
import time
import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ── Load artifacts saved by the notebook ─────────────────────────────────────
with open("model_artifacts.pkl", "rb") as f:
    artifacts = pickle.load(f)

launch_perf_df       = artifacts["launch_perf_df"]
feature_table        = artifacts["feature_table"]
auc                  = artifacts["auc"]
price_elasticity     = artifacts["price_elasticity"]
influencer_uplift_factor = artifacts["influencer_uplift_factor"]
hydration_momentum   = artifacts["hydration_momentum"]

# ── Constants ─────────────────────────────────────────────────────────────────
MONTH_MAP = {
    "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4,
    "May": 5, "Jun": 6, "Jul": 7, "Aug": 8,
    "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12,
}

SEASONAL_INDEX = {
    1: 1.05, 2: 1.02, 3: 1.08, 4: 1.10, 5: 1.12, 6: 1.07,
    7: 0.95, 8: 0.98, 9: 1.03, 10: 1.15, 11: 1.30, 12: 0.75,
}


# ── Helpers ───────────────────────────────────────────────────────────────────
def find_nearest_skus(characteristics: str, search_terms: str, n: int = 5):
    """Score every row in launch_perf_df by keyword overlap and return top-n."""
    keywords = (
        characteristics.lower().replace(",", " ").split()
        + search_terms.lower().split()
    )
    scores = []
    for _, row in launch_perf_df.iterrows():
        haystack = " ".join(
            [
                str(row.get("artikel_name", "")),
                str(row.get("product", "")),
                str(row.get("flavour", "")),
                str(row.get("target_group", "")),
            ]
        ).lower()
        score = sum(1 for kw in keywords if kw in haystack)
        scores.append((score, row))
    scores.sort(key=lambda x: x[0], reverse=True)
    return [r for _, r in scores[:n]]


# ── Core forecast function ────────────────────────────────────────────────────
def generate_forecast(
    product_name: str,
    characteristics: str,
    search_terms: str,
    unit_price: float,
    month_name: str,
    collab_mode: str,
):
    logs = []
    ts = lambda: time.strftime("%H:%M:%S")

    logs.append(f"[{ts()}] [USER]    EXECUTING FORECAST FOR {product_name.upper()}...")
    logs.append(f"[{ts()}] [AI]      ANALYZING CHARACTERISTICS: {characteristics.upper()}...")

    # Nearest-SKU matching
    nearest_skus = find_nearest_skus(characteristics, search_terms, n=5)
    if nearest_skus:
        best = nearest_skus[0]
        logs.append(
            f"[{ts()}] [MATCH]   NEAREST HISTORICAL SKU: "
            f"{best.get('product', 'N/A')} {best.get('flavour', '')}"
        )
    else:
        nearest_skus = (
            launch_perf_df.sort_values("total_quantity", ascending=False)
            .head(5)
            .to_dict("records")
        )
        logs.append(f"[{ts()}] [MATCH]   USING TOP VOLUME SKUs AS REFERENCE")

    logs.append(f"[{ts()}] [COMPUTE] PROCESSING ENSEMBLE MODELS...")

    # Dynamic factors
    seasonal_factor  = SEASONAL_INDEX.get(MONTH_MAP.get(month_name, 11), 1.0)
    avg_price        = launch_perf_df["uvp"].mean()
    dyn_price_factor = (unit_price / avg_price) ** price_elasticity if avg_price > 0 else 1.0
    dyn_inf_factor   = influencer_uplift_factor if collab_mode == "Co-Creation" else 1.0

    # Model 1 — Baseline (nearest-SKU average × factors)
    ref_qty = np.mean([r["total_quantity"] for r in nearest_skus])
    m1 = ref_qty * dyn_price_factor * dyn_inf_factor * seasonal_factor

    # Model 2 — XGBoost propensity sum (re-scaled)
    scale    = dyn_price_factor * dyn_inf_factor * seasonal_factor
    m2_total = feature_table["p_buy_model2"].sum() * scale
    m2_new   = (
        feature_table.loc[feature_table["is_new_customer"] == 1, "p_buy_model2"].sum()
        * scale
    )

    # Model 3 — Scenario (hydration momentum × factors)
    m3 = m2_total * hydration_momentum * dyn_inf_factor * dyn_price_factor * seasonal_factor

    # Blended ensemble (50 / 30 / 20)
    blended  = int(round(0.50 * m1 + 0.30 * m2_total + 0.20 * m3))
    new_cust = int(round(m2_new))

    logs.append(f"[{ts()}] [MODEL1]  BASELINE  → {int(round(m1))} units")
    logs.append(f"[{ts()}] [MODEL2]  XGBOOST   → {int(round(m2_total))} units  (AUC: {auc:.2f})")
    logs.append(f"[{ts()}] [MODEL3]  SCENARIO  → {int(round(m3))} units")
    logs.append(f"[{ts()}] [FINAL]   BLENDED   → {blended} units | NEW CUSTOMERS: {new_cust}")

    # Ensemble bar chart
    names  = ["BASELINE", "XGBOOST", "SCENARIO", "BLENDED"]
    values = [int(round(m1)), int(round(m2_total)), int(round(m3)), blended]
    confidence = round(min(0.99, 0.70 + 0.06 * len(nearest_skus)), 2)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(x=names[:3], y=values[:3], marker_color="#b0bec5", name="Base Models")
    )
    fig.add_trace(
        go.Bar(x=[names[3]], y=[values[3]], marker_color="#1565C0", name="Weighted Result")
    )
    fig.update_layout(
        title=dict(
            text=f"Ensemble Comparison Analysis  |  CONFIDENCE SCORE: {confidence}",
            font=dict(size=13),
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        bargap=0.35,
        height=380,
        yaxis=dict(showgrid=True, gridcolor="#f0f0f0"),
        legend=dict(orientation="h", yanchor="bottom", y=-0.28, xanchor="center", x=0.5),
        margin=dict(l=30, r=30, t=70, b=40),
    )

    # Reference SKU table
    avg_ref  = launch_perf_df["total_quantity"].mean()
    ref_rows = []
    for i, row in enumerate(nearest_skus, start=1):
        pct = ((row["total_quantity"] / avg_ref) - 1) * 100 if avg_ref > 0 else 0
        ref_rows.append(
            {
                "SKU ID":    f"S-{i:03d} Reference",
                "Product":   f"{row.get('product', 'N/A')} – {row.get('flavour', '')}",
                "Base Adj.": f"{'+' if pct >= 0 else ''}{pct:.0f}%",
            }
        )
    ref_df = pd.DataFrame(ref_rows)

    return blended, new_cust, fig, "\n".join(logs), ref_df


# ── Gradio UI ─────────────────────────────────────────────────────────────────
CSS = """
body, .gradio-container {
    font-family: 'Inter', sans-serif !important;
    background: #f4f6f9 !important;
}
.section-label {
    font-size: .68rem;
    font-weight: 700;
    color: #888;
    letter-spacing: .10em;
    text-transform: uppercase;
    margin-bottom: 6px;
}
#telemetry-box textarea {
    background: #1a1a2e !important;
    color: #7ec8e3 !important;
    font-family: monospace !important;
    font-size: .78rem !important;
}
"""

with gr.Blocks(css=CSS, title="Demand Forecasting Ensemble") as demo:

    # ── Header ────────────────────────────────────────────────────────────────
    gr.HTML(
        """
        <div style='background:#fff; border-bottom:1px solid #e0e0e0;
                    padding:14px 24px; display:flex; align-items:center;
                    justify-content:space-between; border-radius:8px 8px 0 0;'>
          <div>
            <div style='font-size:1.15rem; font-weight:700; color:#111;'>
              📦 Demand Forecasting Ensemble
            </div>
            <div style='font-size:.7rem; color:#aaa; letter-spacing:.06em;'>
              MASTER THESIS RESEARCH PROTOTYPE V1.2
            </div>
          </div>
          <div style='display:flex; gap:28px; align-items:center;'>
            <span style='color:#1565C0; font-size:.82rem; font-weight:600;'>
              ⚡ Model Status: <b>Active</b>
            </span>
            <span style='color:#aaa; font-size:.75rem;'>REF: THESIS_2024_0422</span>
          </div>
        </div>
        """
    )

    with gr.Row(equal_height=False):

        # ── Left: input parameters ────────────────────────────────────────────
        with gr.Column(scale=1, min_width=250):
            gr.HTML("<div class='section-label'>⚙ Input Parameters</div>")
            inp_product = gr.Textbox(
                label="PRODUCT NAME", value="NEW PERFORMANCE SNACK"
            )
            inp_chars = gr.Textbox(
                label="CHARACTERISTICS", value="HIGH PROTEIN, VEGAN, COCONUT"
            )
            inp_terms = gr.Textbox(
                label="KEY SEARCH TERMS", value="protein vegan cookie"
            )
            with gr.Row():
                inp_price = gr.Number(label="UNIT PRICE (€)", value=32.9, minimum=0.1)
                inp_month = gr.Dropdown(
                    label="MONTH INDEX", choices=list(MONTH_MAP.keys()), value="Nov"
                )
            inp_collab = gr.Radio(
                label="COLLABORATION MODE",
                choices=["Co-Creation", "Standard"],
                value="Co-Creation",
            )
            btn = gr.Button("Generate Forecast ›", variant="primary", size="lg")

        # ── Center: output metrics + chart ────────────────────────────────────
        with gr.Column(scale=2):
            with gr.Row():
                with gr.Column():
                    gr.HTML(
                        "<div class='section-label'>📦 Estimate. Blended Volume</div>"
                    )
                    out_blended = gr.Number(label="", value=0, interactive=False)
                with gr.Column():
                    gr.HTML(
                        "<div class='section-label'>👤 New Customer Proj.</div>"
                    )
                    out_new_cust = gr.Number(label="", value=0, interactive=False)
            out_chart = gr.Plot(label="")

        # ── Right: telemetry + reference set ──────────────────────────────────
        with gr.Column(scale=1, min_width=270):
            gr.HTML("<div class='section-label'>⌨ Process Telemetry</div>")
            out_log = gr.Textbox(
                label="",
                lines=9,
                max_lines=12,
                interactive=False,
                placeholder="Waiting for forecast run...",
                elem_id="telemetry-box",
            )
            gr.HTML(
                "<div class='section-label' style='margin-top:14px;'>"
                "📋 Model Reference Set</div>"
            )
            out_ref = gr.DataFrame(
                label="",
                headers=["SKU ID", "Product", "Base Adj."],
                interactive=False,
                wrap=True,
            )

    # ── Footer ────────────────────────────────────────────────────────────────
    gr.HTML(
        """
        <div style='text-align:center; font-size:.68rem; color:#bbb; padding:10px 0 4px;'>
          © 2024 Predictive Analytics Laboratory &nbsp;|&nbsp;
          <span style='color:#4caf50;'>● Kernel Online</span> &nbsp;|&nbsp;
          Build: 4.2.0-STABLE
        </div>
        """
    )

    # ── Wire button → forecast function ──────────────────────────────────────
    btn.click(
        fn=generate_forecast,
        inputs=[inp_product, inp_chars, inp_terms, inp_price, inp_month, inp_collab],
        outputs=[out_blended, out_new_cust, out_chart, out_log, out_ref],
    )

if __name__ == "__main__":
    demo.launch(share=True, debug=False)
