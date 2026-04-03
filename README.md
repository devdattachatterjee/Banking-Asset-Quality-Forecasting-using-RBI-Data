# 🏦 Banking Asset Quality: Macro-Financial Stress Testing Framework

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://banking-asset-quality-forecasting-using-rbi-data-badxve4dyejjf.streamlit.app)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/devdattachatterjee/Banking-Asset-Quality-Forecasting-using-RBI-Data/blob/main/Banking_Asset_Quality_Master_Notebook.ipynb)
[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📌 Project Overview

The Indian banking sector's health is one of the most critical indicators of the country's macroeconomic stability. Non-Performing Assets (NPAs) — loans where borrowers have stopped making interest or principal payments — represent the single biggest risk to the financial system. When NPAs spiral out of control, as they did during the 2015–18 twin-balance-sheet crisis in India, it triggers a credit freeze, investment decline, and ultimately a slowdown in economic growth.

This project builds a **Dynamic Macro-Financial Stress Testing Framework** that enables risk analysts, regulators, and banking professionals to:

1. **Forecast** Gross NPA ratios for Scheduled Commercial Banks using historical RBI data
2. **Stress-test** the banking system under adverse economic scenarios (recession, credit crunch, NPA persistence)
3. **Quantify tail risk** using Monte Carlo simulation to compute 95% Value-at-Risk (VaR)
4. **Explain** model predictions using SHAP, ensuring interpretability for regulatory use

The framework is deployed as an interactive Streamlit dashboard, making it accessible to non-technical stakeholders in real time.

---

## 🎯 Problem Statement

> *"Can historical balance-sheet dynamics and credit stress persistence predict future Gross NPA trends in Indian Scheduled Commercial Banks — and can we quantify the worst-case loss under a given macro shock?"*

Traditional bank stress tests are conducted annually by central banks using complex internal models. This project democratises that capability by building a data-driven, open-source equivalent using publicly available RBI data, with results interpretable enough for both regulators and the public.

---

## 📊 Data Source

| Attribute | Detail |
|---|---|
| **Source** | Reserve Bank of India (RBI) — Statistical Tables Relating to Banks in India |
| **Dataset** | Gross and Net NPAs of Scheduled Commercial Banks (Bank Group-wise) |
| **Frequency** | Annual |
| **Coverage** | FY2000 – FY2024 (25 years, 6 bank groups) |
| **Bank Group Used** | Scheduled Commercial Banks (broadest, most representative) |
| **Variables** | Gross Advances, Net Advances, Gross NPA Amount, Gross NPA %, Net NPA Amount, Net NPA % |

The dataset captures **nearly three decades** of Indian banking history, including:
- The NPA build-up phase (2012–2018)
- The Asset Quality Review shock (2015–16)
- The peak NPA crisis (FY2018: ~14.5%)
- The recovery and recapitalisation era (2018–2024)

---

## 🧠 Methodology

### The Analytics "Risk Funnel"

```
 Raw RBI Data (Excel)
        │
        ▼
 Data Cleaning & Parsing
 (multi-header Excel → tidy CSV)
        │
        ▼
 Exploratory Data Analysis
 (GNPA trends, pairplots, correlation heatmaps)
        │
        ▼
 Feature Engineering
 (YoY growth rates, GNPA lag-1, GNPA lag-2, Net NPA %)
        │
        ▼
 Model Development (sklearn Pipelines)
 Linear Regression │ Random Forest │ Gradient Boosting
        │
        ▼
 Temporal Evaluation (75/25 time-aware split)
 MAE · R² · Actual vs Predicted plots
        │
        ▼
 Model Explainability (SHAP)
 Feature importance · Directional impact
        │
        ▼
 ARIMA Baseline Benchmark
 (validates that feature engineering adds value)
        │
        ┌──────────────────┐
        ▼                  ▼
 Deterministic         Monte Carlo
 Stress Engine         Simulation (1000+ runs)
 (Sensitivity)         (95% VaR / Tail Risk)
        │                  │
        └────────┬──────────┘
                 ▼
        Streamlit Dashboard
        (Live Interactive App)
```

---

## ⚙️ Feature Engineering

Five features were engineered to capture the **temporal dynamics of credit stress**:

| Feature | Description | Why It Matters |
|---|---|---|
| `Gross_Advances_Growth` | YoY % change in gross credit | Rapid credit expansion is a leading indicator of future NPA build-up |
| `Net_Advances_Growth` | YoY % change in net credit | Captures net credit momentum excluding provisions |
| `GNPA_lag1` | GNPA ratio of previous year | NPAs exhibit strong path-dependence — stressed banks stay stressed |
| `GNPA_lag2` | GNPA ratio of two years prior | Captures multi-year stress persistence and resolution cycles |
| `Net_NPA_Percent_Advances` | Net NPA as % of advances | Measures the adequacy of provisioning coverage |

> A **time-aware train-test split (75/25, no shuffle)** was used throughout to prevent data leakage — a critical requirement in any financial forecasting task.

---

## 🤖 Models

Three models were implemented inside **scikit-learn Pipelines** (StandardScaler → Estimator):

| Model | Architecture | Purpose |
|---|---|---|
| **Linear Regression** | OLS regression | Interpretable baseline; tests linearity of macro-financial relationships |
| **Random Forest** | 300 trees, ensemble | Captures non-linear interactions between credit cycle variables |
| **Gradient Boosting** | 300 estimators, lr=0.05, depth=3 | Boosted trees; typically strongest on tabular financial data |

Additionally, an **ARIMA(1,1,1)** model was implemented as a classical time-series benchmark to validate that feature engineering with balance-sheet variables meaningfully outperforms naive extrapolation.

---

## 📈 Results

| Model | MAE | R² |
|---|---|---|
| **Linear Regression** | **~0.35%** | **~0.988** |
| Gradient Boosting | ~2.6% | ~0.65 |
| Random Forest | ~2.8% | ~0.48 |
| ARIMA(1,1,1) baseline | ~8.7% | — |

**Key finding:** The Linear Regression pipeline with engineered lag features achieves near-perfect R² (~0.988), dramatically outperforming both tree-based models and the ARIMA baseline. This confirms that **NPA dynamics in Indian banking are highly path-dependent** — the best predictor of this year's NPA ratio is last year's NPA ratio, combined with credit growth momentum.

---

## 🔍 Explainability: SHAP Analysis

SHAP (SHapley Additive exPlanations) was applied to the Random Forest model to understand **feature-level contributions** to each prediction:

- **`GNPA_lag1`** consistently dominates as the most influential feature — confirming that NPA stress in India exhibits strong year-over-year persistence
- **`Net_NPA_Percent_Advances`** captures provisioning adequacy — higher net NPA signals that banks have not written off enough
- **Credit growth features** have directional impact: aggressive credit expansion in prior years negatively affects asset quality

This level of explainability is essential for regulatory compliance and audit trails in real-world risk management applications.

---

## 🖥️ Streamlit Application

The live dashboard enables non-technical users to perform stress testing without writing any code.

**🔗 Live App:** [Click here to open the dashboard](https://banking-asset-quality-forecasting-using-rbi-data-badxve4dyejjf.streamlit.app)

### Features

#### 1. Deterministic Stress Testing
Users apply four economic shocks via sliders:
- **Gross Credit Growth Shock** — simulates a credit crunch or recession (negative values)
- **Net Credit Growth Shock** — net credit momentum under stress
- **NPA Stress Multiplier** — models worsening or improving NPA persistence (>1 = crisis deepens)
- **Net NPA Ratio Multiplier** — models provisioning deterioration or improvement

The model instantly produces a **stressed GNPA forecast** and classifies the result as:
- 🟢 **STABLE** — GNPA below 7%
- 🟡 **WATCHLIST** — GNPA between 7–10%
- 🔴 **CRITICAL** — GNPA above 10%

#### 2. Monte Carlo Simulation (Tail Risk / VaR)
- 500–5,000 stochastic economic paths are generated using Gaussian perturbations around the stressed scenario
- The **95th percentile** of simulated GNPA outcomes is reported as the **Value-at-Risk**
- This answers: *"What is the worst-case outcome we should be prepared for with 95% confidence?"*
- Results are visualised as a probability density plot with the VaR threshold marked

---

## 💼 Real-World Applications

This framework is directly analogous to tools used by:

| Organisation | Use Case |
|---|---|
| **Reserve Bank of India** | Annual Macro Stress Testing Reports (Financial Stability Report) |
| **Individual Commercial Banks** | Internal Capital Adequacy Assessment Process (ICAAP) under Basel III |
| **Credit Rating Agencies** | Forward-looking bank creditworthiness assessments |
| **Investment Analysts** | Screening banking stocks under macro stress scenarios |
| **Finance Ministry** | Estimating recapitalisation requirements for PSBs under adverse conditions |

The RBI's own stress testing framework (published in the Financial Stability Report) uses a similar regression-based approach with macro variables. This project makes an equivalent methodology openly accessible, reproducible, and interactive.

---

## 🛠️ Tech Stack

| Category | Technology |
|---|---|
| **Language** | Python 3.10 |
| **Data Manipulation** | pandas, NumPy |
| **Machine Learning** | scikit-learn (Pipeline, StandardScaler, LinearRegression, RandomForestRegressor, GradientBoostingRegressor) |
| **Explainability** | SHAP (TreeExplainer) |
| **Time Series** | statsmodels (ARIMA) |
| **Visualisation** | matplotlib, seaborn |
| **Web Application** | Streamlit |
| **Deployment** | Streamlit Community Cloud |
| **Notebook Environment** | Google Colab |
| **Version Control** | GitHub |

---

## 📂 Repository Structure

```
Banking-Asset-Quality-Forecasting-using-RBI-Data/
│
├── 📓 Banking_Asset_Quality_Master_Notebook.ipynb   ← Complete end-to-end Colab notebook
├── 🖥️  app.py                                        ← Streamlit application (deployed)
├── 📊 cleaned_bank_npas.csv                          ← Processed RBI data (all bank groups)
├── 📁 Gross and Net NPAs of Scheduled...xlsx         ← Raw RBI source data
└── 📄 README.md                                      ← This file
```

---

## 🚀 Running the Project

### Option 1: Google Colab (Recommended — zero setup)

Click the badge at the top of this README, or use this direct link:

```
https://colab.research.google.com/github/devdattachatterjee/Banking-Asset-Quality-Forecasting-using-RBI-Data/blob/main/Banking_Asset_Quality_Master_Notebook.ipynb
```

Upload the Excel file when prompted, then run all cells top-to-bottom.

### Option 2: Live Streamlit App

No installation needed — open the deployed dashboard directly:

```
https://banking-asset-quality-forecasting-using-rbi-data-badxve4dyejjf.streamlit.app
```

### Option 3: Run Locally

```bash
# Clone the repository
git clone https://github.com/devdattachatterjee/Banking-Asset-Quality-Forecasting-using-RBI-Data.git
cd Banking-Asset-Quality-Forecasting-using-RBI-Data

# Install dependencies
pip install streamlit pandas numpy matplotlib seaborn scikit-learn shap statsmodels openpyxl

# Run the Streamlit app
streamlit run app.py
```

---

## 🔑 Key Concepts Demonstrated

This project covers the following MLOps and Data Science concepts as part of the curriculum:

- ✅ **End-to-end ML pipeline** — from raw data ingestion to deployed application
- ✅ **Feature engineering** — temporal lags, growth rates, domain-driven variable creation
- ✅ **sklearn Pipelines** — encapsulation of preprocessing and modelling steps
- ✅ **Temporal train-test splitting** — preventing data leakage in time-series settings
- ✅ **Model benchmarking** — systematic comparison of ML vs classical statistical approaches
- ✅ **Model explainability** — SHAP values for regulatory-grade interpretability
- ✅ **Stochastic simulation** — Monte Carlo methods for tail-risk quantification
- ✅ **Model deployment** — Streamlit app with CI/CD via GitHub → Streamlit Cloud
- ✅ **Documentation** — reproducible research with rationale at every step

---

## 📖 Business Context: Why This Problem Matters

India's banking sector faced an existential crisis between 2015 and 2018. Gross NPA ratios for Scheduled Commercial Banks climbed from ~4.3% in FY2014 to a peak of **~14.5% in FY2018** — the worst in two decades. This represented over ₹10 lakh crore (approximately $120 billion) in stressed assets, triggering:

- A near-collapse of credit growth in the economy
- A requirement for ₹2.11 lakh crore in government recapitalisation of public sector banks
- Multiple bank failures and forced mergers (PMC Bank, Yes Bank, Lakshmi Vilas Bank)
- A prolonged investment slump and economic slowdown

The **Asset Quality Review (AQR)** conducted by the RBI in 2015-16 forced banks to recognise hidden NPAs, creating a sudden shock to reported figures. This project's stress-testing framework would have allowed institutions to model this scenario in advance — asking: *"What happens to our NPA ratio if credit growth halves and last year's stressed loans keep deteriorating?"*

Since 2018, aggressive resolution through the **Insolvency and Bankruptcy Code (IBC)**, bank recapitalisation, and tighter underwriting standards have reduced the GNPA ratio to approximately **2.7% by FY2024** — a dramatic recovery that this model's lag features also capture.

---

## 👤 Author

**Devdatta Chatterjee**
