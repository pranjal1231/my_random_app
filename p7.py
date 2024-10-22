import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC  
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('output4.csv')

# Define features for water requirement prediction
features_water = ['Soil Moisture', 'Temperature', 'rainfall', 'Air humidity (%)']
X_water = data[features_water]
y_water = pd.cut(data['Watering Required'], bins=[-float('inf'), 0, 20, float('inf')],
                 labels=['No water', 'Minimal water', 'Maximum water'])

# Train-test split for water requirement
X_train_water, X_test_water, y_train_water, y_test_water = train_test_split(X_water, y_water, test_size=0.2, random_state=42)

# Define and train the SVM model for water requirement prediction
svm_model = SVC()
svm_model.fit(X_train_water, y_train_water)

# Define features for soil acidity prediction
scaler = StandardScaler()
X_acidity = data[["ph", "N", "P", "K"]]
y_acidity = data[["Acidity status"]]

# Train-test split for soil acidity
X_train_acidity, X_test_acidity, y_train_acidity, y_test_acidity = train_test_split(X_acidity, y_acidity, test_size=0.2, random_state=42)
X_train_acidity = scaler.fit_transform(X_train_acidity)
X_test_acidity = scaler.transform(X_test_acidity)

# Define and train the KNN model for soil acidity prediction
knn_model = KNeighborsClassifier(n_neighbors=5, weights='distance', p=2)
knn_model.fit(X_train_acidity, y_train_acidity)

# Prepare data for disease risk prediction
X_disease = data[["Air humidity (%)", "Soil Humidity", "Temperature", "Wind gust (Km/h)"]]
y_disease = data["Risk of Disease"]

# Encode risk of disease into categorical risk status
def risk_status(val):
    return 'Low Risk' if val < 95 else 'High Risk'

y_disease = y_disease.apply(risk_status)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_disease)

# Train-test split for disease risk prediction
X_train_disease, X_test_disease, y_train_disease, y_test_disease = train_test_split(X_disease, y_encoded, test_size=0.3, random_state=42)

# Train Random Forest model for disease risk prediction
rf_model = RandomForestClassifier()
rf_model.fit(X_train_disease, y_train_disease)

# Streamlit app interface
st.title("Soil Water Requirement, Acidity, and Disease Risk Prediction")

# Input fields for Water Requirement and Soil Acidity Prediction
st.write("""### Enter the soil parameters below to predict water requirement, soil acidity, and disease risk.""")

# Create a single column for inputs and prediction button
input_col = st.columns(1)
with input_col[0]:
    # Input fields for Water Requirement Prediction
    soil_moisture = st.number_input('Soil Moisture (%)', min_value=0, max_value=100, value=50)
    temperature = st.number_input('Temperature (°C)', min_value=-10, max_value=50, value=25)
    rainfall = st.number_input('Rainfall (mm)', min_value=0, max_value=500, value=100)
    air_humidity = st.number_input('Air Humidity (%)', min_value=0, max_value=100, value=60)

    # Input fields for Soil Acidity Prediction
    ph = st.number_input('pH', min_value=0.0, max_value=14.0, value=7.0)
    n = st.number_input('Nitrogen (N)', min_value=0, max_value=100, value=50)
    p = st.number_input('Phosphorous (P)', min_value=0, max_value=100, value=50)
    k = st.number_input('Potassium (K)', min_value=0, max_value=100, value=50)

    # Prediction for all models
    if st.button('Predict All'):
        # Water Requirement Prediction
        input_data_water = pd.DataFrame({
            'Soil Moisture': [soil_moisture],
            'Temperature': [temperature],
            'rainfall': [rainfall],
            'Air humidity (%)': [air_humidity]
        })
        water_prediction = svm_model.predict(input_data_water)[0]
        st.write(f"### Predicted Water Requirement: **{water_prediction}**")

        # Soil Acidity Prediction
        input_data_acidity = pd.DataFrame({
            'ph': [ph],
            'N': [n],
            'P': [p],
            'K': [k]
        })
        input_data_acidity_scaled = scaler.transform(input_data_acidity)
        acidity_prediction = knn_model.predict(input_data_acidity_scaled)[0]
        st.write(f"### Predicted Soil Acidity Status: **{acidity_prediction}**")

        # Disease Risk Prediction
        input_data_disease = pd.DataFrame({
            "Air humidity (%)": [air_humidity],
            "Soil Humidity": [soil_moisture],  # Adjusted according to the input fields
            "Temperature": [temperature],
            "Wind gust (Km/h)": [rainfall]  # Using rainfall as a placeholder for wind gust
        })
        disease_prediction = rf_model.predict(input_data_disease)[0]
        st.write(f"### Predicted Disease Risk Status: **{label_encoder.inverse_transform([disease_prediction])[0]}**")

# Create three columns for performance metrics
metrics_col1, metrics_col2 = st.columns(2)

# Water Requirement performance metrics
with metrics_col1:
    # Show model performance for Water Requirement prediction
    y_pred_water = svm_model.predict(X_test_water)
    accuracy_water = accuracy_score(y_test_water, y_pred_water)
    st.write(f"**Water Requirement Accuracy**: {accuracy_water:.2f}")

    # Confusion Matrix for Water Requirement
    cm_water = confusion_matrix(y_test_water, y_pred_water)
    fig_water, ax_water = plt.subplots(figsize=(4, 3))  # Smaller figure size
    sns.heatmap(cm_water, annot=True, fmt='d', cmap='Blues', ax=ax_water)
    ax_water.set_title('Confusion Matrix for Water Requirement')
    ax_water.set_ylabel('True Label')
    ax_water.set_xlabel('Predicted Label')
    st.pyplot(fig_water)
    
with metrics_col2:
    
     # Classification Report for Water Requirement
    st.write("### Classification Report for Water Requirement")
    report_water = classification_report(y_test_water, y_pred_water, output_dict=True)
    report_water_df = pd.DataFrame(report_water).transpose()
    st.dataframe(report_water_df)

# Soil Acidity performance metrics
with metrics_col1:
    # Show model performance for Soil Acidity prediction
    y_pred_acidity = knn_model.predict(X_test_acidity)
    accuracy_acidity = accuracy_score(y_test_acidity, y_pred_acidity)
    st.write(f"**Soil Acidity Accuracy**: {accuracy_acidity:.2f}")
    
    # Classification Report for Soil Acidity
    st.write("### Classification Report for Soil Acidity")
    report_acidity = classification_report(y_test_acidity, y_pred_acidity, output_dict=True)
    report_acidity_df = pd.DataFrame(report_acidity).transpose()
    st.dataframe(report_acidity_df)

with metrics_col2:
    # Confusion Matrix for Soil Acidity
    cm_acidity = confusion_matrix(y_test_acidity, y_pred_acidity)
    fig_acidity, ax_acidity = plt.subplots(figsize=(4, 3))  # Smaller figure size
    sns.heatmap(cm_acidity, annot=True, fmt='d', cmap='Blues', ax=ax_acidity)
    ax_acidity.set_title('Confusion Matrix for Soil Acidity')
    ax_acidity.set_ylabel('True Label')
    ax_acidity.set_xlabel('Predicted Label')
    st.pyplot(fig_acidity)

# Disease Risk performance metrics
with metrics_col1:
    # Show model performance for Disease Risk prediction
    y_pred_disease = rf_model.predict(X_test_disease)
    accuracy_disease = accuracy_score(y_test_disease, y_pred_disease)
    st.write(f"**Disease Risk Accuracy**: {accuracy_disease:.2f}")
    
    # Confusion Matrix for Disease Risk
    cm_disease = confusion_matrix(y_test_disease, y_pred_disease)
    fig_disease, ax_disease = plt.subplots(figsize=(4, 3))  # Smaller figure size
    sns.heatmap(cm_disease, annot=True, fmt='d', cmap='Blues', ax=ax_disease)
    ax_disease.set_title('Confusion Matrix for Disease Risk')
    ax_disease.set_ylabel('True Label')
    ax_disease.set_xlabel('Predicted Label')
    st.pyplot(fig_disease)

with metrics_col2:
    
    # Classification Report for Disease Risk
    st.write("### Classification Report for Disease Risk")
    report_disease = classification_report(y_test_disease, y_pred_disease, output_dict=True)
    report_disease_df = pd.DataFrame(report_disease).transpose()
    st.dataframe(report_disease_df)

# File upload feature
uploaded_file = st.file_uploader("Upload a CSV file for batch prediction")

if uploaded_file is not None:
    batch_data = pd.read_csv(uploaded_file)

    # Check if required columns are present
    required_columns_water = ['Soil Moisture', 'Temperature', 'rainfall', 'Air humidity (%)']
    required_columns_acidity = ['ph', 'N', 'P', 'K']
    required_columns_disease = ["Air humidity (%)", "Soil Humidity", "Temperature", "Wind gust (Km/h)"]

    # Validate the uploaded batch data
    if all(col in batch_data.columns for col in required_columns_water + required_columns_acidity + required_columns_disease):
        # Prepare the data for predictions
        water_inputs = batch_data[required_columns_water]
        acidity_inputs = batch_data[required_columns_acidity]
        disease_inputs = batch_data[required_columns_disease]

        # Make predictions for water requirement
        water_predictions = svm_model.predict(water_inputs)

        # Scale acidity inputs for the KNN model
        acidity_inputs_scaled = scaler.transform(acidity_inputs)
        acidity_predictions = knn_model.predict(acidity_inputs_scaled)

        # Make predictions for disease risk
        disease_predictions = rf_model.predict(disease_inputs)

        # Create a results DataFrame
        results = batch_data.copy()
        results['Water Requirement Prediction'] = water_predictions
        results['Soil Acidity Prediction'] = acidity_predictions
        results['Disease Risk Prediction'] = label_encoder.inverse_transform(disease_predictions)

        # Display results
        st.write("### Batch Prediction Results")
        st.dataframe(results)

        # Option to download results
        csv = results.to_csv(index=False)
        st.download_button(label="Download Results as CSV", data=csv, file_name='batch_predictions.csv', mime='text/csv')

    else:
        st.error("Uploaded CSV does not contain the required columns for predictions.")

