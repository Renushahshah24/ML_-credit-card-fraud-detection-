import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix

# Load the trained model


@st.cache_resource
def load_model():
    model = joblib.load('fraud_detection_model.pkl')
    return model

# Load the model
clf = load_model()

st.title('Credit Card Fraud Detection')
st.write('This app detects potentially fraudulent credit card transactions')
uploaded_file = st.file_uploader("Upload a CSV file for prediction", type="csv")

if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
    
    # Display the uploaded data
    st.write("Uploaded Data:")
    st.dataframe(input_df)
    
    # Make predictions
    if st.button('Predict Fraud'):
        predictions = clf.predict(input_df)
        prediction_proba = clf.predict_proba(input_df)
        
        # Add predictions to dataframe
        result_df = input_df.copy()
        result_df['is_fraud_prediction'] = predictions
        result_df['fraud_probability'] = prediction_proba[:, 1]  # Probability of class 1 (fraud)
        
        # Display results
        st.write("Prediction Results:")
        st.dataframe(result_df)
        
        # Count of fraud vs non-fraud
        fraud_count = sum(predictions)
        st.write(f"Number of fraudulent transactions detected: {fraud_count} out of {len(predictions)}")
        
        # Download results
        csv = result_df.to_csv(index=False)
        st.download_button(
            label="Download predictions as CSV",
            data=csv,
            file_name='fraud_predictions.csv',
            mime='text/csv',
        )
