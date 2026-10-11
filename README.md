# 🛡️ CrediGuard AI - Automated Credit Risk & Loan Default Assessment Engine

An end-to-end AI-powered system for automating credit risk analysis and predicting loan defaults. Built with a relational SQL backend and machine learning pipeline.

### 🚀 Key Features
- **SQL Relational Database:** Structured schema with `customers` and `loan_records`
- **Data Pipeline:** Automated data cleaning and feature engineering with Pandas
- **ML Model:** Handles imbalanced data using SMOTE and predicts default probability
- **Interactive UI:** Real-time risk assessment with Streamlit

### 🛠️ Tech Stack
Python, Streamlit, SQLite, Pandas, NumPy, Scikit-Learn, Imbalanced-Learn (SMOTE)


### ⚙️ How to Run
```bash
pip install -r requirements.txt
streamlit run app.py