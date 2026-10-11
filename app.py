import sqlite3
from imblearn.over_sampling import SMOTE
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import streamlit as st


@st.cache_resource
def load_trained_model():
  conn = sqlite3.connect('crediguard.db')
  cursor = conn.cursor()

  cursor.execute('DROP TABLE IF EXISTS customers')
  cursor.execute('DROP TABLE IF EXISTS loan_records')

  cursor.execute("""
    CREATE TABLE customers (
        customer_id INT PRIMARY KEY,
        age INT,
        annual_income DECIMAL(10, 2),
        credit_score INT
    )
    """)

  cursor.execute("""
    CREATE TABLE loan_records (
        loan_id INT PRIMARY KEY,
        customer_id INT,
        loan_amount DECIMAL(10, 2),
        interest_rate DECIMAL(5, 2),
        term_months INT,
        is_default INT,
        FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
    )
    """)

  customers_data = [
      (101, 28, 650000.00, 750),
      (102, 35, 420000.00, 610),
      (103, 42, 1200000.00, 810),
      (104, 24, 300000.00, 580),
      (105, 30, 550000.00, 690),
      (106, 45, 250000.00, 520),
      (107, 26, 800000.00, 780),
      (108, 38, 490000.00, 630),
      (109, 29, 700000.00, 720),
      (110, 50, 200000.00, 490),
      (111, 32, 900000.00, 790),
      (112, 25, 350000.00, 590),
      (113, 33, 400000.00, 600),
      (114, 40, 310000.00, 530),
  ]
  loans_data = [
      (501, 101, 200000.00, 10.5, 36, 0),
      (502, 102, 350000.00, 14.0, 48, 1),
      (503, 103, 500000.00, 8.5, 24, 0),
      (504, 104, 150000.00, 16.0, 12, 1),
      (505, 105, 300000.00, 11.0, 36, 0),
      (506, 106, 280000.00, 18.0, 24, 1),
      (507, 107, 400000.00, 9.0, 36, 0),
      (508, 108, 320000.00, 13.5, 48, 0),
      (509, 109, 450000.00, 10.0, 36, 0),
      (510, 110, 180000.00, 20.0, 24, 1),
      (511, 111, 600000.00, 8.0, 48, 0),
      (512, 112, 220000.00, 15.0, 12, 1),
      (513, 113, 300000.00, 17.0, 36, 1),
      (514, 114, 250000.00, 19.0, 24, 1),
  ]

  cursor.executemany(
      'INSERT INTO customers VALUES (?, ?, ?, ?)', customers_data
  )
  cursor.executemany(
      'INSERT INTO loan_records VALUES (?, ?, ?, ?, ?, ?)', loans_data
  )
  conn.commit()

  query = """
    SELECT c.age, c.annual_income, c.credit_score, 
           l.loan_amount, l.interest_rate, l.term_months, l.is_default
    FROM customers c
    JOIN loan_records l ON c.customer_id = l.customer_id
    """
  df = pd.read_sql(query, conn)
  conn.close()

  monthly_income = df['annual_income'] / 12
  monthly_interest = df['interest_rate'] / (12 * 100)
  df['estimated_emi'] = (
      df['loan_amount']
      * monthly_interest
      * (1 + monthly_interest) ** df['term_months']
  ) / ((1 + monthly_interest) ** df['term_months'] - 1)
  df['dti_ratio'] = df['estimated_emi'] / monthly_income
  df['income_per_age'] = np.log1p(df['annual_income'] / df['age'])

  features = [
      'age',
      'annual_income',
      'credit_score',
      'loan_amount',
      'interest_rate',
      'term_months',
      'estimated_emi',
      'dti_ratio',
      'income_per_age',
  ]
  X = df[features]
  y = df['is_default']

  smote = SMOTE(k_neighbors=1, random_state=42)
  X_res, y_res = smote.fit_resample(X, y)

  model = RandomForestClassifier(n_estimators=100, random_state=42)
  model.fit(X_res, y_res)

  return model


model = load_trained_model()

st.set_page_config(
    page_title='CrediGuard AI', page_icon='💳', layout='centered'
)

st.title('🛡️ CrediGuard AI')
st.subheader('Automated Credit Risk & Loan Default Assessment Engine')
st.markdown(
    'This AI system uses a relational SQL backend, Pandas data pipelines,'
    ' and SMOTE-balanced Machine Learning to evaluate loan risk.'
)
st.divider()

col1, col2 = st.columns(2)

with col1:
  age = st.number_input('Customer Age', min_value=18, max_value=100, value=30)
  annual_income = st.number_input(
      'Annual Income (₹)', min_value=50000, max_value=10000000, value=600000
  )
  credit_score = st.number_input(
      'Credit Score', min_value=300, max_value=900, value=720
  )

with col2:
  loan_amount = st.number_input(
      'Loan Amount Requested (₹)', min_value=10000, max_value=5000000, value=300000
  )
  interest_rate = st.number_input(
      'Interest Rate (%)', min_value=1.0, max_value=30.0, value=11.5
  )
  term_months = st.selectbox(
      'Loan Term (Months)', options=[12, 24, 36, 48, 60], index=2
  )

st.divider()

if st.button('Evaluate Loan Risk 🚀', use_container_width=True):
  m_income = annual_income / 12
  m_interest = interest_rate / (12 * 100)
  est_emi = (
      loan_amount * m_interest * (1 + m_interest) ** term_months
  ) / ((1 + m_interest) ** term_months - 1)
  dti = est_emi / m_income
  inc_age = np.log1p(annual_income / age)

  input_data = pd.DataFrame([[
      age,
      annual_income,
      credit_score,
      loan_amount,
      interest_rate,
      term_months,
      est_emi,
      dti,
      inc_age,
  ]], columns=[
      'age',
      'annual_income',
      'credit_score',
      'loan_amount',
      'interest_rate',
      'term_months',
      'estimated_emi',
      'dti_ratio',
      'income_per_age',
  ])

  prediction = model.predict(input_data)[0]
  probability = model.predict_proba(input_data)[0][1] * 100

  if prediction == 1 or probability > 50:
    st.error(
        f'⚠️ **High Risk: Loan Default Likely!** (Risk Probability:'
        f' {probability:.2f}%)'
    )
    st.markdown(
        '*Recommendation:* Reject loan application or require a higher'
        ' guarantor/collateral.'
    )
  else:
    st.success(
        f'✅ **Low Risk: Loan Approved!** (Default Probability:'
        f' {probability:.2f}%)'
    )
    st.markdown(
        '*Recommendation:* Customer financial profile is strong. Safe to'
        ' disburse loan.'
    )