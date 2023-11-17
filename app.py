import streamlit as st
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the pre-trained machine learning model and scaler
with open('model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Create a Streamlit app
st.title("Churn Prediction App")
st.write("Enter customer attributes to predict churn")

# Define input fields for user input
InternetService = st.selectbox("Internet Service", ['DSL', 'Fiber optic'])
OnlineSecurity = st.selectbox("Online Security", ['No', 'Yes'])
TechSupport = st.selectbox("Tech Support", ['No', 'Yes'])
contract = st.selectbox("Contract", ['Month-to-month', 'One year', 'Two year'])
PaymentMethod = st.selectbox("Payment Method", ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
tenure = st.number_input("Tenure", min_value=0, value=100)
MonthlyCharges = st.number_input("Monthly Charges", min_value=0, value=1000)
TotalCharges = st.number_input("Total Charges", min_value=0, value=1000)

# Create a button to trigger the prediction
if st.button("Predict Churn"):
    # Prepare the input data for prediction
    input_data = {
        "InternetService_Fiber optic": [1 if InternetService == 'Fiber optic' else 0],
        "OnlineSecurity_No": [1 if OnlineSecurity == 'No' else 0],
        "TechSupport_No": [1 if TechSupport == 'No' else 0],
        "Contract_Month-to-month": [1 if contract == 'Month-to-month' else 0],
        "PaymentMethod_Electronic check": [1 if PaymentMethod == 'Electronic check' else 0],
        "tenure": [tenure],
        "MonthlyCharges": [MonthlyCharges],
        "TotalCharges": [TotalCharges]
    }

    input_df = pd.DataFrame(input_data)
    scaled_input_data = scaler.transform(input_df)

    # Make the prediction
    predicted_churn = loaded_model.predict(scaled_input_data)

     # Calculate the confidence factor (assuming binary classification)
    confidence_factor = predicted_churn.squeeze()

    # Display the prediction and confidence factor
    st.write(f"Predicted Churn: {int(round(float(confidence_factor)))}")
    st.write(f"Confidence Factor: {confidence_factor:.2f}")

    # Display the prediction
    if confidence_factor > 0.5:
        st.warning('Churn: Yes')
    else:
        st.success('Churn: No')

# Create a reset button to clear the input fields
if st.button("Reset"):
    st.experimental_rerun()
