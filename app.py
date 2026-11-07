import joblib
import pandas as pd
import streamlit as st
from pathlib import Path

st.set_page_config(
    page_title="Telecom Customer Retention Assistant",
    page_icon="📊",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "telecom_churn_model.pkl"
FEATURES_PATH = BASE_DIR / "model_features.pkl"

st.title("Telecom Customer Retention Assistant")
st.write(
    "Enter a customer's service details to estimate churn risk "
    "and view practical retention suggestions."
)

@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    features = joblib.load(FEATURES_PATH)
    return model, features

try:
    model, model_features = load_artifacts()
except FileNotFoundError as exc:
    st.error(
        f"Required model file not found: {exc.filename}. "
        "Place telecom_churn_model.pkl and model_features.pkl "
        "in the same folder as app.py."
    )
    st.stop()
except Exception as exc:
    st.error(f"Could not load the model files: {exc}")
    st.stop()

with st.form("customer_details"):
    st.subheader("Customer details")

    c1, c2, c3 = st.columns(3)

    with c1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior_citizen = st.selectbox("Senior citizen", [0, 1],
                                      format_func=lambda x: "Yes" if x == 1 else "No")
        partner = st.selectbox("Has a partner?", ["Yes", "No"])
        dependents = st.selectbox("Has dependents?", ["Yes", "No"])
        tenure = st.slider("Tenure (months)", 0, 72, 12)

    with c2:
        phone_service = st.selectbox("Phone service", ["Yes", "No"])
        multiple_lines = st.selectbox(
            "Multiple lines", ["No", "Yes", "No phone service"]
        )
        internet_service = st.selectbox(
            "Internet service", ["DSL", "Fiber optic", "No"]
        )
        online_security = st.selectbox(
            "Online security", ["Yes", "No", "No internet service"]
        )
        online_backup = st.selectbox(
            "Online backup", ["Yes", "No", "No internet service"]
        )
        device_protection = st.selectbox(
            "Device protection", ["Yes", "No", "No internet service"]
        )

    with c3:
        tech_support = st.selectbox(
            "Tech support", ["Yes", "No", "No internet service"]
        )
        streaming_tv = st.selectbox(
            "Streaming TV", ["Yes", "No", "No internet service"]
        )
        streaming_movies = st.selectbox(
            "Streaming movies", ["Yes", "No", "No internet service"]
        )
        contract = st.selectbox(
            "Contract", ["Month-to-month", "One year", "Two year"]
        )
        paperless_billing = st.selectbox("Paperless billing", ["Yes", "No"])
        payment_method = st.selectbox(
            "Payment method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )

    c4, c5 = st.columns(2)
    with c4:
        monthly_charges = st.number_input(
            "Monthly charges ($)", min_value=0.0, max_value=200.0,
            value=70.0, step=1.0
        )
    with c5:
        total_charges = st.number_input(
            "Total charges ($)", min_value=0.0, max_value=20000.0,
            value=840.0, step=10.0
        )

    submitted = st.form_submit_button("Predict churn risk", type="primary")

if submitted:
    customer = {
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }

    missing = [col for col in model_features if col not in customer]
    if missing:
        st.error(
            "The saved feature list contains columns not covered by this form: "
            + ", ".join(missing)
        )
        st.stop()

    input_df = pd.DataFrame(
        [[customer[col] for col in model_features]],
        columns=model_features
    )

    try:
        prediction = model.predict(input_df)[0]
        classes = list(model.classes_)
        yes_index = classes.index("Yes")
        churn_probability = float(model.predict_proba(input_df)[0][yes_index])
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        st.stop()

    if churn_probability >= 0.60:
        risk = "High"
    elif churn_probability >= 0.30:
        risk = "Medium"
    else:
        risk = "Low"

    st.divider()
    st.subheader("Prediction result")

    r1, r2 = st.columns(2)
    with r1:
        st.metric("Estimated churn probability", f"{churn_probability:.1%}")
    with r2:
        st.metric("Churn risk", risk)

    st.progress(churn_probability)

    if risk == "High":
        st.error("High risk: consider prioritising this customer for retention outreach.")
    elif risk == "Medium":
        st.warning("Medium risk: monitor this customer and consider a targeted offer.")
    else:
        st.success("Low risk: continue regular customer engagement.")

    suggestions = []
    if contract == "Month-to-month":
        suggestions.append("Offer a suitable longer-term contract incentive.")
    if tenure < 12:
        suggestions.append("Provide an onboarding check-in and early-tenure support.")
    if monthly_charges >= 70:
        suggestions.append("Review the customer's plan and explain lower-cost suitable options.")
    if payment_method == "Electronic check":
        suggestions.append("Explain the convenience of automatic payment options.")
    if tech_support == "No" and internet_service != "No":
        suggestions.append("Offer information about available technical support plans.")
    if not suggestions:
        suggestions.append("Maintain service quality and use regular satisfaction check-ins.")

    st.subheader("Suggested retention actions")
    for item in suggestions:
        st.write(f"• {item}")

    st.caption(
        "Risk bands are illustrative business thresholds: Low < 30%, "
        "Medium 30–59.9%, High ≥ 60%. They are not separately trained model classes."
    )
