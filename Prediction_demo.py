import io
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import gzip
from pathlib import Path

st.set_page_config(page_title="Stacked Ensemble Tester", page_icon="⚡", layout="wide")

@st.cache_resource(show_spinner=False)
def load_models():
    models_dir = Path(__file__).parent / "Models" 

    def load_gz_model(filename):
        with gzip.open(models_dir / filename, 'rb') as f:
            return joblib.load(f)

    lgbm_model = load_gz_model("lgbm_base_model.pkl.gz")
    xgb_model = load_gz_model("xgboost_base_model.pkl.gz")
    cat_model = load_gz_model("catboost_base_model.pkl.gz")
    ridge_model = load_gz_model("ridge_base_model.pkl.gz")
    meta_model = load_gz_model("meta_model.pkl.gz")

    return lgbm_model, xgb_model, cat_model, ridge_model, meta_model

def gather_input_data(expected_n_cols) -> pd.DataFrame:
    st.header("Input Data")
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    
    if uploaded_file is None:
        st.warning("Please upload a CSV file to proceed.")
        st.stop()  # stops the app run here; nothing below executes yet

    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error reading the file: {e}")
        st.stop()  # stop so downstream code doesn't run with a bad df

    # Drop first column if it's 'ID' (case-insensitive, trims spaces)
    if df.shape[1] > 0:
        first_col = df.columns[0]
        if first_col.strip().lower() == "id":
            df = df.drop(columns=[first_col])

    # Validate column count
    n_cols = df.shape[1]
    n_rows = df.shape[0]
    if n_cols != expected_n_cols:
        st.error(
            f"Expected {expected_n_cols} feature columns, "
            f"but found {n_cols}. Please upload the correct CSV."
        )
        st.stop()  # force user to re-upload
    elif n_rows == 0:
        st.error("The uploaded CSV file is empty. Please upload a valid file.")
        st.stop()


    st.write("Data Preview:")
    st.dataframe(df.head())
    return df

def download_template():
    st.subheader("Download Input Template")
    st.write("Download the template if you want to try the model with your own data. Remember that the data should be prepared in the same format as the training data.")
    # Define your expected columns
    column_names = [
    "Component1_fraction", "Component2_fraction", "Component3_fraction", "Component4_fraction", "Component5_fraction",
    "Component1_Property1", "Component2_Property1", "Component3_Property1", "Component4_Property1", "Component5_Property1",
    "Component1_Property2", "Component2_Property2", "Component3_Property2", "Component4_Property2", "Component5_Property2",
    "Component1_Property3", "Component2_Property3", "Component3_Property3", "Component4_Property3", "Component5_Property3",
    "Component1_Property4", "Component2_Property4", "Component3_Property4", "Component4_Property4", "Component5_Property4",
    "Component1_Property5", "Component2_Property5", "Component3_Property5", "Component4_Property5", "Component5_Property5",
    "Component1_Property6", "Component2_Property6", "Component3_Property6", "Component4_Property6", "Component5_Property6",
    "Component1_Property7", "Component2_Property7", "Component3_Property7", "Component4_Property7", "Component5_Property7",
    "Component1_Property8", "Component2_Property8", "Component3_Property8", "Component4_Property8", "Component5_Property8",
    "Component1_Property9", "Component2_Property9", "Component3_Property9", "Component4_Property9", "Component5_Property9",
    "Component1_Property10", "Component2_Property10", "Component3_Property10", "Component4_Property10", "Component5_Property10"
]

    # Create empty DataFrame with just headers
    template_df = pd.DataFrame(columns=column_names)

    # Convert to CSV
    csv_data = template_df.to_csv(index=False)

    # Create a download button
    st.download_button(
        label="📥 Download CSV Template",
        data=csv_data,
        file_name="input_template.csv",
        mime="text/csv"
    )

st.title("Sustainable Avaiation Fuel Property Predictor")

download_template()
x_test = gather_input_data(55)

from src.data import remove_nulls, add_weighted_features, compute_entropy, add_weighted_mean_properties, add_constraint_aware_features

x_test = remove_nulls(x_test)
n_rows = x_test.shape[0]
if n_rows == 0: 
    st.error("The input data is empty after removing nulls. Please check your input.")
    st.stop()  # stop so downstream code doesn't run with an empty df
x_test = add_weighted_features(x_test)
x_test['blend_entropy'] = x_test.apply(compute_entropy, axis=1)
x_test = add_weighted_mean_properties(x_test)
x_test = add_constraint_aware_features(x_test)
to_drop = [c for c in x_test.columns
           if "component4" in c.lower() and "weighted" in c.lower()]
x_test = x_test.drop(columns=to_drop)
lgbm_model, xgb_model, cat_model, ridge_model, meta_model = load_models()

st.header("Model Predictions")

stacked_preds = meta_model.predict(np.hstack([
    lgbm_model.predict(x_test),
    ridge_model.predict(x_test),
    cat_model.predict(x_test),
    xgb_model.predict(x_test)
]))

# Store as DataFrame and assign column names
stacked_pred_df = pd.DataFrame(stacked_preds,columns = ["Blend_Property1", "Blend_Property2", "Blend_Property3",
"Blend_Property4", "Blend_Property5", "Blend_Property6", "Blend_Property7", "Blend_Property8",
"Blend_Property9", "Blend_Property10"])

downloadtype = st.write("Do you want prediction to be displayed here or downlaoded as CSV?")

option = st.selectbox(
    "Do you want prediction to be displayed here or downlaoded as CSV?",
    ("Select","In-site", "CSV")
)

if option == "CSV":
    csv = stacked_pred_df.to_csv(index=False)
    st.download_button(
        label="Download Predictions as CSV",
        data=csv,
        file_name="predictions.csv",
        mime="text/csv"
    )
elif option == "In-site":
    st.write("Predictions:")
    st.dataframe(stacked_pred_df)




