# 🏦 Banking Asset Quality: Macro-Financial Stress Testing Framework

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_svg.svg)](https://banking-asset-quality-forecasting-using-rbi-data-badxve4dyejjf.streamlit.app)

## 🚀 Live Project
**Interactive Risk Dashboard:** [View Live App](https://banking-asset-quality-forecasting-using-rbi-data-badxve4dyejjf.streamlit.app)

---

## 📌 Project Executive Summary
How sensitive is the Indian banking sector to a sudden recession or a spike in inflation? This project transitions from traditional static forecasting to **Dynamic Risk Modeling**. 

Using historical **Reserve Bank of India (RBI)** statistical data, I engineered a predictive engine for **Gross Non-Performing Assets (GNPA)**. The framework allows users to apply synthetic "shocks" to the economy to observe how asset quality degrades under pressure. It is designed to mimic the stress-testing protocols used by central banks and internal risk committees to assess financial stability.

---

## 🧠 The Analytics "Risk Funnel"
The project utilizes a two-stage modeling approach to quantify credit risk and capital vulnerability:

### 1. Deterministic Stress Testing (Sensitivity Analysis)
Users can manually adjust macroeconomic stressors—**GDP Growth Shocks, Inflation Spikes, and Interest Rate hikes**—to see the immediate impact on the GNPA ratio. This identifies the "Linear Sensitivity" of bank balance sheets to specific economic levers.

### 2. Stochastic Monte Carlo Simulation (Tail-Risk)
Because the future is non-linear and uncertain, the system executes **1,000+ Stochastic Simulations**. By randomly sampling from the distribution of economic variables, it calculates the **95% Value-at-Risk (VaR)**.
* **Objective:** To identify the "Worst Case Scenario" (the 5% tail-event) where asset quality could spiral out of control, providing a probabilistic view of potential capital erosion.

---

## 🛠️ Technical Tech Stack & Methodology
* **Core Modeling:** Random Forest Regressor (Ensemble Learning) to capture complex, non-linear macro-financial linkages.
* **Simulation:** Monte Carlo Methods & Stochastic Gaussian Sampling.
* **Visualization:** Seaborn & Matplotlib for Risk Distribution plotting and Probability Density Analysis.
* **Frontend:** Streamlit Cloud for real-time model inference and interactive "What-If" analysis.
* **Data Source:** RBI Statistical Tables Relating to Banks in India.

---

## 📂 Key Features
* **Real-Time Delta Tracking:** Compares "Baseline" vs. "Stressed" GNPA forecasts instantly to show risk escalation.
* **Risk Sentiment Logic:** Automatically categorizes results into *Stable*, *Watchlist*, or *Critical* based on industry-standard threshold breach analysis.
* **Stochastic Distribution:** Visualizes the probability density of potential NPA outcomes to assist in macroprudential oversight.

---

## ⚙️ How to Run Locally
1. Clone the repository:
   ```bash
   git clone [https://github.com/devdattachatterjee/banking-asset-quality-forecasting-using-rbi-data.git](https://github.com/devdattachatterjee/banking-asset-quality-forecasting-using-rbi-data.git)
