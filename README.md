# Banking Asset Quality Forecasting using RBI Data

## 📌 Project Overview
This project focuses on forecasting **banking asset quality deterioration** in India using officially published data from the **Reserve Bank of India (RBI)**. The objective is to model and explain the evolution of **Gross Non-Performing Assets (GNPA)** for Scheduled Commercial Banks by leveraging balance-sheet indicators and historical stress persistence.

The project emphasizes **real-world data, explainable machine learning, and finance-domain reasoning**, avoiding synthetic datasets or black-box approaches.

---

## 🎯 Business Problem
Non-Performing Assets (NPAs) are a critical indicator of banking system health. Rising NPAs signal:
- Credit risk accumulation
- Weakening underwriting standards
- Potential systemic stress

This project answers the question:

**Can historical balance-sheet dynamics and past stress levels help forecast future GNPA trends in Indian banks?**

---

## 📊 Data Source
- **Source:** Reserve Bank of India (RBI)
- **Dataset:** Gross and Net NPAs of Scheduled Commercial Banks (Bank Group-wise)
- **Frequency:** Annual
- **Bank Group Used:** Scheduled Commercial Banks

The data was cleaned and structured manually to ensure accuracy and reproducibility.

---

## 🧾 Key Variables

### Target Variable
- **Gross_NPA_Percent_Advances**  
  (Gross NPAs as a percentage of Gross Advances)

### Explanatory Variables
- Gross Advances Growth (YoY)
- Net Advances Growth (YoY)
- Lagged GNPA (1-year and 2-year lags)
- Net NPA Percentage

---

## ⚙️ Feature Engineering
To capture banking stress dynamics:
- **Growth rates** were computed using year-on-year percentage change
- **Lagged GNPA features** were created to model persistence of credit stress
- Time-aware splitting was used to prevent data leakage

---

## 🤖 Models Used
Three models were implemented and compared:

1. **Linear Regression** – Baseline interpretability check  
2. **Random Forest Regressor** – Non-linear ensemble model  
3. **Gradient Boosting Regressor** – Strong performer for tabular financial data  

All models were implemented using **scikit-learn Pipelines**.

---

## 📈 Model Evaluation
Models were evaluated using:
- Mean Absolute Error (MAE)
- R² Score
- Visual comparison of actual vs predicted GNPA

A time-based train–test split was used to respect temporal ordering.

---

## 🔍 Explainability
To ensure interpretability:
- **SHAP (SHapley Additive exPlanations)** was used
- Feature importance and directional impact were analyzed
- Results confirm that lagged GNPA and advances growth are dominant drivers

---

## 🛠️ Tech Stack
- Python
- pandas, NumPy
- scikit-learn
- matplotlib, seaborn
- SHAP

---

## 📂 Project Structure
