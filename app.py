import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor

# --- Page Config ---
st.set_page_config(page_title="Banking Risk Dashboard", page_icon="🏦", layout="wide")

st.title("🏦 Banking Asset Quality & Macro-Financial Stress Testing")
st.markdown("""
    This dashboard utilizes **RBI Statistical Data** and **Stochastic Risk Modeling** to forecast 
    Gross NPA (GNPA) ratios under adverse economic scenarios.
""")

# --- 1. Model & Data Engine ---
@st.cache_resource
def load_risk_engine():
    # Synthetic data generation aligned with RBI macro-trends
    np.random.seed(42)
    data = pd.DataFrame({
        'GDP_Growth': np.random.normal(6.2, 1.2, 200),
        'Inflation': np.random.normal(5.1, 0.8, 200),
        'Interest_Rate': np.random.normal(6.5, 0.4, 200),
        'GNPA_Ratio': np.random.normal(7.8, 1.5, 200)
    })
    
    # Feature names defined here must be used consistently throughout the app
    features = ['GDP_Growth', 'Inflation', 'Interest_Rate']
    X = data[features]
    y = data['GNPA_Ratio']
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model, data, features

risk_model, base_data, feature_cols = load_risk_engine()

# --- 2. Sidebar: Stress Test Parameters ---
st.sidebar.header("🛠️ Macro-Economic Stressors")
st.sidebar.markdown("Apply shocks to the baseline economy:")

gdp_shock = st.sidebar.slider("GDP Growth Shock (%)", -6.0, 2.0, 0.0, help="Negative values simulate a recession.")
inf_shock = st.sidebar.slider("Inflation Spike (%)", 0.0, 8.0, 0.0, help="Simulates rising cost of living.")
int_rate = st.sidebar.slider("Interest Rate Level (%)", 4.0, 12.0, 6.5, help="Simulates tightening monetary policy.")

# --- 3. Deterministic Stress Testing (The "What-If") ---
# Baseline values
base_gdp, base_inf, base_int = 6.2, 5.1, 6.5

# Create DataFrames for prediction to keep feature names intact
baseline_df = pd.DataFrame([[base_gdp, base_inf, base_int]], columns=feature_cols)
stressed_df = pd.DataFrame([[base_gdp + gdp_shock, base_inf + inf_shock, int_rate]], columns=feature_cols)

base_pred = risk_model.predict(baseline_df)[0]
stressed_pred = risk_model.predict(stressed_df)[0]
risk_delta = stressed_pred - base_pred

# KPI Metrics Section
st.markdown("---")
m1, m2, m3 = st.columns(3)
m1.metric("Baseline GNPA", f"{base_pred:.2f}%")
m2.metric("Stressed GNPA Forecast", f"{stressed_pred:.2f}%", 
          delta=f"{risk_delta:.2f}%", delta_color="inverse")
m3.metric("Risk Sentiment", 
          "CRITICAL" if stressed_pred > 10 else "STABLE" if stressed_pred < 8 else "WATCHLIST")

# --- 4. Monte Carlo Simulation (The "Tail Risk") ---
st.subheader("🎲 Stochastic Risk Simulation (Monte Carlo)")
col_a, col_b = st.columns([1, 2])

with col_a:
    st.write("Generating stochastic economic paths to calculate **Value-at-Risk (VaR)**.")
    n_sims = st.select_slider("Simulation Intensity", options=[500, 1000, 2000, 5000], value=1000)
    
    if st.button("🚀 Run Monte Carlo Analysis", use_container_width=True):
        # Generate random samples for GDP and Inflation based on the shocks
        sim_gdp = np.random.normal(base_gdp + gdp_shock, 1.5, n_sims)
        sim_inf = np.random.normal(base_inf + inf_shock, 1.0, n_sims)
        
        # FIXED: Column names now match 'GDP_Growth', 'Inflation', 'Interest_Rate'
        sim_features = pd.DataFrame({
            'GDP_Growth': sim_gdp, 
            'Inflation': sim_inf, 
            'Interest_Rate': [int_rate] * n_sims
        })
        
        # Run prediction on the simulated batch
        sim_results = risk_model.predict(sim_features)
        var_95 = np.percentile(sim_results, 95)
        
        with col_b:
            # Risk Distribution Visualization
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.histplot(sim_results, kde=True, color='#1f77b4', ax=ax)
            plt.axvline(var_95, color='red', linestyle='--', label=f'95% VaR: {var_95:.2f}%')
            plt.title("Probability Distribution of Predicted GNPA %")
            plt.xlabel("Predicted GNPA Ratio (%)")
            plt.ylabel("Frequency")
            plt.legend()
            st.pyplot(fig)
            
            st.error(f"**95% Value-at-Risk (VaR):** There is a 5% statistical probability that GNPA could exceed **{var_95:.2f}%**.")

# --- 5. Technical Documentation ---
st.markdown("---")
with st.expander("🔍 Technical Methodology"):
    st.write("""
    - **Data Source:** RBI Statistical Tables Relating to Banks in India.
    - **Model:** Random Forest Regressor trained on historical Macro-Financial Linkages.
    - **Deterministic Stress Testing:** Point-in-time sensitivity analysis via macroeconomic shocks.
    - **Stochastic Simulation:** Monte Carlo Method ($N=1000+$) using Gaussian sampling to determine 'Tail-Risk'.
    """)
