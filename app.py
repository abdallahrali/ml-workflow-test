import joblib
import pandas as pd
import streamlit as st

st.title("House Price Predictor (Workflow Test)")


@st.cache_resource
def load_model():
    return joblib.load("models/house_price_prediction_model.joblib")


model = load_model()

area = st.number_input(
    "Area (m²)", min_value=30.0, max_value=500.0, value=120.0, step=5.0
)
bedrooms = st.slider("Bedrooms", min_value=1, max_value=6, value=3)

if st.button("Predict Price"):
    X_new = pd.DataFrame([{"area": area, "bedrooms": bedrooms}])
    pred = model.predict(X_new)[0]
    st.success(f"Estimated Price: ${pred:,.1f}k")
