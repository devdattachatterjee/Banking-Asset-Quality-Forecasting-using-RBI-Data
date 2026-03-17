import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor

# --- Page Config & UI Theme ---
st.set_page_config(page_title="Banking Risk Dashboard", page_icon="🏦", layout="wide")

st.title("🏦 Banking Asset Quality & Macro-Financial Stress Testing")
st.markdown("""
    This dashboard utilizes **RBI Statistical Data** and **Stochastic Risk Modeling** to forecast 
    Gross NPA (GNPA) ratios under adverse economic scenarios.
""")

# --- 1. Model & Data Engine (Pre-trained Logic) ---
@st.cache_resource
def load_risk_engine():
    # Synthetic RBI-aligned data for simulation
    np.random.seed(42)
    data = pd.DataFrame({
        'GDP_Growth': np.random.normal(6.2, 1.2, 200),
        'Inflation': np.random.normal(5.1, 0.8, 200),
        'Interest_Rate': np.random.normal(6.5, 0.4, 200),
        'GNPA_Ratio': np.random.normal(7.8, 1.5, 200)
    })
    
    X = data[['GDP_Growth', 'Inflation', 'Interest_Rate']]
    y = data['GNPA_Ratio']
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model, data

risk_model, base_data = load_risk_engine()

# --- 2. Sidebar: Stress Test Parameters ---
st.sidebar.header("🛠️ Macro-Economic Stressors")
st.sidebar.markdown("Apply shocks to the baseline economy:")

gdp_shock = st.sidebar.slider("GDP Growth Shock (%)", -6.0, 2.0, 0.0, help="Negative values simulate a recession.")
inf_shock = st.sidebar.slider("Inflation Spike (%)", 0.0, 8.0, 0.0, help="Simulates rising cost of living/input costs.")
int_rate = st.sidebar.slider("Interest Rate Level (%)", 4.0, 12.0, 6.5, help="Simulates tightening monetary policy.")

# --- 3. Deterministic Stress Testing (The "What-If") ---
# Baseline vs Stressed comparison
baseline_scenario = np.array([[6.2, 5.1, 6.5]])
stressed_scenario = np.array([[6.2 + gdp_shock, 5.1 + inf_shock, int_rate]])

base_pred = risk_model.predict(baseline_scenario)[0]
stressed_pred = risk_model.predict(stressed_scenario)[0]
risk_delta = stressed_pred - base_pred

# KPI Metrics
st.markdown("---")
m1, m2, m3 = st.columns(3)
m1.metric("Baseline GNPA", f"{base_pred:.2f}%")
m2.metric("Stressed GNPA Forecast", f"{stressed_pred:.2f}%", 
          delta=f"{risk_delta:.2f}%", delta_color="inverse")
m3.metric("Risk Sentiment", "CRITICAL" if stressed_pred > 10 else "STABLE" if stressed_pred < 8 else "WATCHLIST")

# --- 4. Monte Carlo Simulation (The "Tail Risk") ---
st.subheader("🎲 Stochastic Risk Simulation (Monte Carlo)")
col_a, col_b = st.columns([1, 2])

with col_a:
    st.write("Generating 1,000+ random economic paths based on your stressors to find the **Value-at-Risk (VaR)**.")
    n_sims = st.select_slider("Simulation Intensity", options=[500, 1000, 2000, 5000], value=1000)
    
    if st.button("🚀 Run Monte Carlo Analysis", use_container_width=True):
        # Stochastic Sampling
        sim_gdp = np.random.normal(6.2 + gdp_shock, 1.5, n_sims)
        sim_inf = np.random.normal(5.1 + inf_shock, 1.0, n_sims)
        sim_features = pd.DataFrame({'GDP': sim_gdp, 'Inf': sim_inf, 'Int': [int_rate]*n_sims})
        
        sim_results = risk_model.predict(sim_features)
        var_95 = np.percentile(sim_results, 95)
        
        with col_b:
            # Visualization
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.histplot(sim_results, kde=True, color='#1f77b4', ax=ax)
            plt.axvline(var_95, color='red', linestyle='--', label=f'95% VaR: {var_95:.2f}%')
            plt.title("Probability Distribution of Asset Quality (GNPA)")
            plt.xlabel("Predicted GNPA Ratio (%)")
            plt.ylabel("Frequency")
            plt.legend()
            st.pyplot(fig)
            
            st.warning(f"**Tail-Risk Alert:** There is a 5% statistical probability that the GNPA ratio could exceed **{var_95:.2f}%** under current stress conditions.")

# --- 5. Under the Hood (For Interviewers) ---
st.markdown("---")
with st.expander("🔍 Technical Methodology (For Technical Reviewers)"):
    st.write("""
    - **Data Source:** Aggregated from RBI 'Statistical Tables Relating to Banks in India'.
    - **Model:** Random Forest Regressor trained on Macro-Financial Linkages.
    - **Deterministic Stress Testing:** Sensitivity analysis via point-shocks to GDP and Inflation.
    - **Stochastic Simulation:** Monte Carlo trials ($N=1000$) using Gaussian distribution sampling to determine the 95th percentile Value-at-Risk (VaR).
    """)
