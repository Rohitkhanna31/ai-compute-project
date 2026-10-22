import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="AI Compute | Data Story", page_icon="◈", layout="wide")

st.markdown("""
<style>
.stApp {background:#0b1020;color:#edf2ff}
[data-testid="stSidebar"] {background:#11182b;border-right:1px solid #26334d}
.block-container {padding:5rem 2rem 3rem 2rem;max-width:100%;overflow-wrap:anywhere}
h1,h2,h3 {color:#f4f7ff!important}
p,label,.stMarkdown {color:#c6d0e3}
[data-testid="stMetric"] {background:linear-gradient(135deg,#18243b,#111a2d);border:1px solid #2b3b59;padding:20px;border-radius:14px}
[data-testid="stMetricLabel"] {color:#aebdd6!important}
[data-testid="stMetricValue"] {color:#76e4cf!important}
div[data-testid="stPlotlyChart"] {background:#11182b;border:1px solid #26334d;border-radius:14px;padding:8px}
.story-card {background:linear-gradient(135deg,#172943,#121b30);border:1px solid #304362;border-radius:16px;padding:22px;margin:8px 0 16px;line-height:1.65}
.eyebrow {color:#76e4cf;font-size:12px;font-weight:700;letter-spacing:2px}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    data = pd.read_csv("all_ai_models.csv")
    data.columns = data.columns.str.strip()
    if "Publication date" in data.columns:
        data["Publication date"] = pd.to_datetime(data["Publication date"], errors="coerce")
    for col in ["Parameters", "Training compute (FLOP)", "Training dataset size (total)"]:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")
    return data

try:
    df = load_data()
except Exception as exc:
    st.error(f"Could not load all_ai_models.csv: {exc}")
    st.info("Make sure app.py and all_ai_models.csv are in the same project folder.")
    st.stop()

required = ["Model", "Publication date", "Parameters", "Training compute (FLOP)", "Training dataset size (total)"]
missing = [col for col in required if col not in df.columns]
if missing:
    st.error("Expected columns not found: " + ", ".join(missing))
    st.stop()

st.sidebar.markdown("## ◈ AI COMPUTE LAB")
st.sidebar.caption("A data-driven exploration")
st.sidebar.divider()
valid_years = df["Publication date"].dt.year.dropna()
selected_years = None
if not valid_years.empty:
    year_min, year_max = int(valid_years.min()), int(valid_years.max())
    selected_years = st.sidebar.slider("Publication year", year_min, year_max, (year_min, year_max))
search = st.sidebar.text_input("Search model", placeholder="e.g. GPT")
selected_domains = []
if "Domain" in df.columns:
    domains = sorted(v for v in df["Domain"].dropna().astype(str).unique() if v.strip())
    selected_domains = st.sidebar.multiselect("Model domain", domains)

filtered = df.copy()
if selected_years:
    filtered = filtered[filtered["Publication date"].dt.year.between(*selected_years)]
if search:
    filtered = filtered[filtered["Model"].astype(str).str.contains(search, case=False, na=False)]
if selected_domains and "Domain" in filtered.columns:
    filtered = filtered[filtered["Domain"].astype(str).isin(selected_domains)]

st.markdown('<p class="eyebrow">EPOCH AI DATA · EXPLORATORY ANALYSIS</p>', unsafe_allow_html=True)
st.title("AI Training Compute")
st.markdown("### Bigger models — or more training data?")
st.markdown("How have model scale, training compute and dataset size changed over time? Explore the evidence, missing values and individual models.")
st.divider()

compute, params, dataset = filtered["Training compute (FLOP)"], filtered["Parameters"], filtered["Training dataset size (total)"]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Models in selection", f"{len(filtered):,}")
c2.metric("Compute reported", f"{compute.notna().sum():,}")
c3.metric("Parameter values", f"{params.notna().sum():,}")
c4.metric("Dataset-size values", f"{dataset.notna().sum():,}")
st.caption("Counts describe the selected rows. Missing values are not treated as zero.")

st.markdown("## 01 / The landscape")
st.markdown('<div class="story-card"><b>Start with the evidence.</b><br>Training compute is the computational work used to train a model. Parameter count describes model scale, while dataset size describes the recorded training data measure. These are related but different dimensions; their relationship must be tested rather than assumed.</div>', unsafe_allow_html=True)

plot_df = filtered.dropna(subset=["Publication date", "Training compute (FLOP)"]).copy()
plot_df = plot_df[plot_df["Training compute (FLOP)"] > 0]
if not plot_df.empty:
    plot_df["Compute (log10 FLOP)"] = np.log10(plot_df["Training compute (FLOP)"])
    plot_df["Year"] = plot_df["Publication date"].dt.year
    yearly = plot_df.groupby("Year").agg(Models=("Model", "count"), Median_compute_log10=("Compute (log10 FLOP)", "median")).reset_index()
    left, right = st.columns(2)
    with left:
        st.markdown("#### Training compute over time")
        st.caption("Each point is a model; vertical axis is log₁₀ FLOP.")
        fig_time = px.scatter(plot_df, x="Publication date", y="Compute (log10 FLOP)", hover_name="Model",
            hover_data={"Training compute (FLOP)": ":.4g", "Publication date": True, "Compute (log10 FLOP)": False},
            labels={"Publication date": "Publication date", "Compute (log10 FLOP)": "Training compute (log₁₀ FLOP)"},
            template="plotly_dark", opacity=0.72)
        fig_time.update_layout(paper_bgcolor="#11182b", plot_bgcolor="#11182b", font_color="#edf2ff", margin=dict(l=20,r=20,t=35,b=20))
        st.plotly_chart(fig_time, use_container_width=True)
    with right:
        st.markdown("#### Typical compute by year")
        st.caption("Median log₁₀ compute among models with reported positive values.")
        fig_year = px.line(yearly, x="Year", y="Median_compute_log10", markers=True,
            hover_data={"Models": True, "Median_compute_log10": ":.3f"},
            labels={"Year":"Publication year","Median_compute_log10":"Median compute (log₁₀ FLOP)"}, template="plotly_dark")
        fig_year.update_traces(line_width=3, marker_size=8)
        fig_year.update_layout(paper_bgcolor="#11182b", plot_bgcolor="#11182b", font_color="#edf2ff", margin=dict(l=20,r=20,t=35,b=20))
        st.plotly_chart(fig_year, use_container_width=True)
else:
    st.info("No valid positive training-compute values in this selection.")

# ---------- CORRELATION FINDINGS ----------
st.markdown("## Evidence / What moves with training compute?")

corr_params = filtered[
    ["Training compute (FLOP)", "Parameters"]
].dropna()
corr_params = corr_params[
    (corr_params["Training compute (FLOP)"] > 0) &
    (corr_params["Parameters"] > 0)
]

corr_data = filtered[
    ["Training compute (FLOP)", "Training dataset size (total)"]
].dropna()
corr_data = corr_data[
    (corr_data["Training compute (FLOP)"] > 0) &
    (corr_data["Training dataset size (total)"] > 0)
]

param_rho = (
    corr_params.corr(method="spearman").iloc[0, 1]
    if len(corr_params) >= 2 else np.nan
)
data_rho = (
    corr_data.corr(method="spearman").iloc[0, 1]
    if len(corr_data) >= 2 else np.nan
)

k1, k2 = st.columns(2)
k1.metric("Compute ↔ Parameters", f"{param_rho:.3f}" if pd.notna(param_rho) else "N/A")
k2.metric("Compute ↔ Dataset size", f"{data_rho:.3f}" if pd.notna(data_rho) else "N/A")

st.caption(
    f"Pairwise sample sizes: parameters = {len(corr_params):,}; "
    f"dataset size = {len(corr_data):,}. Updates with sidebar filters."
)

if pd.notna(param_rho) and pd.notna(data_rho):
    stronger = "parameter count" if param_rho > data_rho else "dataset size"
    st.markdown(
        f"**Current result:** Compute has the stronger Spearman association "
        f"with {stronger} in the selected records. This is correlation, "
        "not proof of causation."
    )


st.markdown("## 02 / Bigger models or more data?")
st.write("Compare reported training compute with parameter count and dataset size. Only rows with valid positive values for both variables appear in each chart.")
a, b = st.columns(2)
with a:
    st.markdown("#### Compute vs parameters")
    param_df = filtered.dropna(subset=["Training compute (FLOP)", "Parameters"]).copy()
    param_df = param_df[(param_df["Training compute (FLOP)"] > 0) & (param_df["Parameters"] > 0)]
    if not param_df.empty:
        param_df["Log10 compute"] = np.log10(param_df["Training compute (FLOP)"])
        param_df["Log10 parameters"] = np.log10(param_df["Parameters"])
        fig_param = px.scatter(param_df, x="Log10 parameters", y="Log10 compute", hover_name="Model",
            hover_data={"Parameters":":,.4g","Training compute (FLOP)":":.4g","Log10 parameters":False,"Log10 compute":False},
            
            labels={"Log10 parameters":"Model size (log₁₀ parameters)","Log10 compute":"Training compute (log₁₀ FLOP)"},
            template="plotly_dark", opacity=0.7)
        fig_param.update_layout(paper_bgcolor="#11182b", plot_bgcolor="#11182b", font_color="#edf2ff", margin=dict(l=20,r=20,t=60,b=20))
        st.plotly_chart(fig_param, use_container_width=True)
        st.caption(f"{len(param_df):,} models with both values reported.")
    else:
        st.info("No rows have both valid compute and parameter values.")

with b:
    st.markdown("#### Compute vs dataset size")
    data_df = filtered.dropna(subset=["Training compute (FLOP)", "Training dataset size (total)"]).copy()
    data_df = data_df[(data_df["Training compute (FLOP)"] > 0) & (data_df["Training dataset size (total)"] > 0)]
    if not data_df.empty:
        data_df["Log10 compute"] = np.log10(data_df["Training compute (FLOP)"])
        data_df["Log10 dataset size"] = np.log10(data_df["Training dataset size (total)"])
        fig_data = px.scatter(data_df, x="Log10 dataset size", y="Log10 compute", hover_name="Model",
            hover_data={"Training dataset size (total)":":,.4g","Training compute (FLOP)":":.4g","Log10 dataset size":False,"Log10 compute":False},
           
            labels={"Log10 dataset size":"Dataset size (log₁₀)","Log10 compute":"Training compute (log₁₀ FLOP)"},
            template="plotly_dark", opacity=0.7)
        fig_data.update_layout(paper_bgcolor="#11182b", plot_bgcolor="#11182b", font_color="#edf2ff", margin=dict(l=20,r=20,t=60,b=20))
        st.plotly_chart(fig_data, use_container_width=True)
        st.caption(f"{len(data_df):,} models with both values reported.")
    else:
        st.info("No rows have both valid compute and dataset-size values.")

st.markdown('<div class="story-card"><b>How to read this section</b><br>An upward pattern indicates an association in the selected records, not proof that one factor caused compute to rise. The two charts can have different sample sizes because the dataset has missing values.</div>', unsafe_allow_html=True)

st.markdown("## 03 / What is missing?")
quality_cols = ["Parameters", "Training compute (FLOP)", "Training dataset size (total)"]
quality = pd.DataFrame({"Measure":quality_cols, "Reported":[filtered[c].notna().sum() for c in quality_cols], "Missing":[filtered[c].isna().sum() for c in quality_cols]})
quality_long = quality.melt(id_vars="Measure", var_name="Availability", value_name="Models")
quality_fig = px.bar(quality_long, x="Measure", y="Models", color="Availability", barmode="group", template="plotly_dark", color_discrete_sequence=["#76e4cf","#f2a65a"])
quality_fig.update_layout(paper_bgcolor="#11182b", plot_bgcolor="#11182b", font_color="#edf2ff", margin=dict(l=20,r=20,t=30,b=20), legend_title_text="")
st.plotly_chart(quality_fig, use_container_width=True)
st.caption("Reported means non-missing, not necessarily independently verified. Consult confidence and notes before drawing strong conclusions.")

# ---------- FINAL DATA STORY ----------
st.markdown("## 03 / What the evidence suggests")

st.markdown(
    f"""
    <div class="story-card">
        <h3>Model scale has the stronger observed association</h3>
        <p>
        Training compute has a Spearman correlation of
        <b>{param_rho:.3f}</b> with parameter count, compared with
        <b>{data_rho:.3f}</b> for recorded training dataset size.
        </p>
        <p>
        This suggests that parameter count has a slightly stronger
        association with training compute in these records. Both
        relationships are positive and strong, and this comparison
        does not establish causation.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.caption(
    "Interpretation applies to the current filters and available observations. "
    "The two correlations use different sample sizes; missing data and other "
    "factors may affect the comparison."
)


st.markdown("## 04 / Explore a model")
model_names = filtered["Model"].dropna().astype(str).unique().tolist()
if model_names:
    chosen = st.selectbox("Choose a model", model_names)
    row = filtered[filtered["Model"].astype(str) == chosen].iloc[0]
    m1, m2, m3 = st.columns(3)
    m1.markdown("**Publication date**")
    m1.write(str(row["Publication date"].date()) if pd.notna(row["Publication date"]) else "Not reported")
    m2.markdown("**Training compute (FLOP)**")
    m2.write(f"{row['Training compute (FLOP)']:,.4g}" if pd.notna(row["Training compute (FLOP)"]) else "Not reported")
    m3.markdown("**Parameters**")
    m3.write(f"{row['Parameters']:,.4g}" if pd.notna(row["Parameters"]) else "Not reported")
    st.markdown("**Training dataset size (as recorded)**")
    st.write(f"{row['Training dataset size (total)']:,.4g}" if pd.notna(row["Training dataset size (total)"]) else "Not reported")
    for col, label in [("Training dataset","Training dataset"),("Confidence","Source confidence field"),("Training compute notes","Training compute notes")]:
        if col in filtered.columns:
            st.markdown(f"**{label}**")
            value = row.get(col)
            st.write(value if pd.notna(value) else "Not reported")
    if "Link" in filtered.columns and pd.notna(row.get("Link")):
        st.markdown(f"[Open model source]({row['Link']})")
else:
    st.info("No models match the current filters.")

st.divider()
st.markdown("## About this analysis")
st.write("Source: Epoch AI, *Data on AI models*. https://epoch.ai/data/ai-models")
st.write("Dataset licensed under Creative Commons Attribution (CC BY 4.0). Credit: Epoch AI. This dashboard is exploratory; it does not establish causation, and missing or uncertain values can affect comparisons.")
st.caption("Method: publication dates and numeric fields are parsed from the CSV. Charts requiring log scales exclude non-positive values and rows missing the variables shown. Filter choices affect displayed results.")
st.caption("Built with Python, Pandas, Plotly and Streamlit.")
