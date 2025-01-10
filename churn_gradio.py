import gradio as gr
import joblib
import pandas as pd

# Load the trained model
model = joblib.load("model/churn_model.pkl")

# Updated predict_churn function
def predict_churn(SeniorCitizen, tenure, MonthlyCharges, TotalCharges, gender, Partner, Dependents, PhoneService,
                  MultipleLines, InternetService, OnlineSecurity, DeviceProtection, TechSupport, StreamingTV,
                  StreamingMovies, Contract, PaperlessBilling, PaymentMethod):
    # Input features as a dictionary
    input_data = {
        'SeniorCitizen': [SeniorCitizen],
        'tenure': [tenure],
        'MonthlyCharges': [MonthlyCharges],
        'TotalCharges': [TotalCharges],
        'gender': [gender],
        'Partner': [Partner],
        'Dependents': [Dependents],
        'PhoneService': [PhoneService],
        'MultipleLines': [MultipleLines],
        'InternetService': [InternetService],
        'OnlineSecurity': [OnlineSecurity],
        'DeviceProtection': [DeviceProtection],
        'TechSupport': [TechSupport],
        'StreamingTV': [StreamingTV],
        'StreamingMovies': [StreamingMovies],
        'Contract': [Contract],
        'PaperlessBilling': [PaperlessBilling],
        'PaymentMethod': [PaymentMethod],
    }

    # Convert to DataFrame
    input_df = pd.DataFrame(input_data)

    # Load the saved feature names and encoder
    feature_names = joblib.load("model/feature_names.pkl")
    le = joblib.load("label_encoder.pkl")

    # Apply label encoding to categorical columns
    for col in input_df.select_dtypes(include='object').columns:
        if col in le:
            input_df[col] = le[col].transform(input_df[col])

    # Reorder columns to match the training data
    input_df = input_df.reindex(columns=feature_names, fill_value=0)

    # Make prediction
    prediction = model.predict(input_df)
    return "Churn" if prediction[0] == 1 else "No Churn"


# Define the Gradio interface
interface = gr.Interface(
    fn=predict_churn,
    inputs=[
        gr.Number(label="Senior Citizen (0 = No, 1 = Yes)"),
        gr.Number(label="Tenure (months)"),
        gr.Number(label="Monthly Charges"),
        gr.Number(label="Total Charges"),
        gr.Textbox(label="Gender (Male/Female)"),
        gr.Textbox(label="Partner (Yes/No)"),
        gr.Textbox(label="Dependents (Yes/No)"),
        gr.Textbox(label="Phone Service (Yes/No)"),
        gr.Textbox(label="Multiple Lines (No/Yes/No phone service)"),
        gr.Textbox(label="Internet Service (DSL/Fiber optic/No)"),
        gr.Textbox(label="Online Security (Yes/No/No internet service)"),
        gr.Textbox(label="Device Protection (Yes/No/No internet service)"),
        gr.Textbox(label="Tech Support (Yes/No/No internet service)"),
        gr.Textbox(label="Streaming TV (Yes/No/No internet service)"),
        gr.Textbox(label="Streaming Movies (Yes/No/No internet service)"),
        gr.Textbox(label="Contract (Month-to-month/One year/Two year)"),
        gr.Textbox(label="Paperless Billing (Yes/No)"),
        gr.Textbox(label="Payment Method (Electronic check/Mailed check/Bank transfer (automatic)/Credit card (automatic))")
    ],
    outputs=gr.Textbox(label="Prediction"),
    title="Telco Customer Churn Prediction",
    description="Input customer details to predict churn likelihood."
)

# Launch the app
interface.launch()
