import streamlit as st
import pandas as pd
import tempfile
from datetime import datetime
from pycaret.regression import (
    setup, compare_models, pull, finalize_model, 
    save_model, predict_model, plot_model
)

st.set_page_config(page_title="ML Studio Demo", layout="wide")
st.title("🏠 House Price Prediction ML Studio")

# -------------------------------
# Step 1. Upload dataset
# -------------------------------
uploaded_file = st.file_uploader("Upload your dataset (CSV/XLSX)", type=["csv", "xls", "xlsx"])

if uploaded_file:
    try:
        if uploaded_file.name.lower().endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()

    st.success(f"Loaded {len(df)} rows and {len(df.columns)} columns")
    st.dataframe(df.head())

    # -------------------------------
    # Step 2. Target selection
    # -------------------------------
    target = st.selectbox("Select target column (what to predict)", df.columns)

    # -------------------------------
    # Step 3. Train models
    # -------------------------------
    if st.button("🚀 Start Training"):
        with st.spinner("Training models... please wait"):
            exp = setup(data=df, target=target, session_id=123, silent=True, verbose=False)
            top_models = compare_models(n_select=6, sort='RMSE')
            results = pull()

        st.subheader("📊 Model Comparison")
        st.dataframe(results)

        # Save results as CSV for download
        tmp_csv = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        results.to_csv(tmp_csv.name, index=False)
        st.download_button(
            label="Download Model Comparison CSV",
            data=open(tmp_csv.name, "rb").read(),
            file_name=f"model_results_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv"
        )

        # Finalize best model
        best = top_models[0] if isinstance(top_models, list) else top_models
        final_model = finalize_model(best)
        save_model(final_model, "best_model")

        st.success("✅ Training complete. Best model saved as best_model.pkl")

        # -------------------------------
        # Step 4. Feature importance
        # -------------------------------
        st.subheader("🔍 Feature Importance")
        st.pyplot(plot_model(final_model, plot='feature', display_format='streamlit'))

        # Save model in session state
        st.session_state["final_model"] = final_model

# -------------------------------
# Step 5. Predictions
# -------------------------------
if "final_model" in st.session_state:
    st.header("📈 Make Predictions")
    st.write("Enter new data manually:")

    with st.form("prediction_form"):
        # Example fields, adjust to your dataset columns
        sq = st.number_input("Square Feet", value=2300)
        beds = st.number_input("Bedrooms", value=3)
        baths = st.number_input("Bathrooms", value=2)
        age = st.number_input("Age (years)", value=10)
        location = st.text_input("Location", "Suburb")

        submitted = st.form_submit_button("Predict")

        if submitted:
            new_data = pd.DataFrame([{
                "SquareFt": sq,
                "Beds": beds,
                "Baths": baths,
                "Age": age,
                "Location": location
            }])
            preds = predict_model(st.session_state["final_model"], data=new_data)
            st.subheader("Prediction Result")
            st.write(preds)
