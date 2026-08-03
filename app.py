import streamlit as st
import joblib
import pandas as pd

# ---------- Page Config ----------
st.set_page_config(
    page_title="Ford Price Predictor",
    page_icon="🚗",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ---------- Custom CSS (Oberlo-inspired: cream + orange) ----------
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Poppins', sans-serif;
    }

    .stApp {
        background-color: #fefefd;
        color: #1b1b1b;
    }

    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        text-align: center;
        color: #1b1b1b;
        margin-bottom: 0.2rem;
    }

    .main-title span {
        color: #fc9d22;
    }

    .subtitle {
        text-align: center;
        color: #6f6f6f;
        font-size: 0.95rem;
        font-weight: 400;
        margin-bottom: 2.3rem;
        letter-spacing: 0.2px;
    }

    div[data-testid="stSelectbox"] label, div[data-testid="stNumberInput"] label {
        color: #474646 !important;
        font-weight: 500;
        font-size: 0.85rem;
        letter-spacing: 0.2px;
    }

    div[data-baseweb="select"] > div, .stNumberInput input {
        background-color: #ffffff !important;
        border: 1.5px solid #e9e9e8 !important;
        border-radius: 8px !important;
        color: #1b1b1b !important;
    }

    div[data-baseweb="select"] > div:hover, .stNumberInput input:hover {
        border-color: #fc9d22 !important;
    }

    .stButton > button {
        width: 100%;
        background-color: #fc9d22;
        color: #ffffff;
        font-weight: 600;
        font-size: 1rem;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 0;
        margin-top: 1.3rem;
        transition: all 0.25s ease;
        letter-spacing: 0.3px;
    }

    .stButton > button:hover {
        background-color: #de6624;
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(252, 157, 34, 0.3);
    }

    .result-card {
        background-color: #fff7ec;
        border: 1.5px solid #fc9d22;
        border-radius: 14px;
        padding: 1.8rem;
        text-align: center;
        margin-top: 1.8rem;
    }

    .result-label {
        color: #929191;
        font-size: 0.82rem;
        font-weight: 500;
        letter-spacing: 0.6px;
        text-transform: uppercase;
        margin-bottom: 0.4rem;
    }

    .result-price {
        font-size: 2.4rem;
        font-weight: 700;
        color: #fc9d22;
    }

    div[data-testid="stExpander"] {
        background-color: #ffffff;
        border: 1.5px solid #e9e9e8 !important;
        border-radius: 10px;
    }

    hr {
        border-color: #e9e9e8;
    }
    </style>
""", unsafe_allow_html=True)

# ---------- Load Model ----------
model = joblib.load('car_price_model.pkl')
label_encoders = joblib.load('label_encoders.pkl')
scaler = joblib.load('scaler.pkl')

# ---------- Header ----------
st.markdown('<div class="main-title">Ford <span>Price</span> Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Estimate the resale value of a used Ford in seconds</div>', unsafe_allow_html=True)

# ---------- Main Inputs ----------
col1, col2 = st.columns(2)

with col1:
    model_name = st.selectbox("Model", label_encoders['model'].classes_)
    mileage = st.number_input("Mileage", min_value=0, value=20000, step=1000)

with col2:
    year = st.number_input("Year", min_value=1996, max_value=2025, value=2018)
    transmission = st.selectbox("Transmission", label_encoders['transmission'].classes_)

fuel_type = st.selectbox("Fuel Type", label_encoders['fuelType'].classes_)

# ---------- Advanced Options ----------
with st.expander("⚙️  Advanced options"):
    tax = st.number_input("Tax (£)", min_value=0, value=150)
    mpg = st.number_input("MPG", min_value=0.0, value=55.0)
    engine_size = st.number_input("Engine Size (L)", min_value=0.0, value=1.0, step=0.1)

st.markdown("<br>", unsafe_allow_html=True)

# ---------- Predict ----------
if st.button("Predict Price"):
    input_df = pd.DataFrame([{
        'model': model_name,
        'year': year,
        'transmission': transmission,
        'mileage': mileage,
        'fuelType': fuel_type,
        'tax': tax,
        'mpg': mpg,
        'engineSize': engine_size
    }])

    for col in ['model', 'transmission', 'fuelType']:
        input_df[col] = label_encoders[col].transform(input_df[col].astype(str))

    numerical_cols = ['year', 'mileage', 'tax', 'mpg', 'engineSize']
    input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])

    input_df = input_df[['model', 'year', 'transmission', 'mileage', 'fuelType', 'tax', 'mpg', 'engineSize']]

    prediction = model.predict(input_df)[0]

    st.markdown(f"""
        <div class="result-card">
            <div class="result-label">Estimated Price</div>
            <div class="result-price">£{prediction:,.0f}</div>
        </div>
    """, unsafe_allow_html=True)
