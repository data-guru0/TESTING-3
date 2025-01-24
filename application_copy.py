from flask import Flask, request, render_template, jsonify
import pickle
import pandas as pd
import numpy as np
from alibi_detect.cd import KSDrift  # Alibi Detect for Drift Detection
from sklearn.preprocessing import StandardScaler
from src.data_processing import RedisFeatureStore  # Assuming your RedisFeatureStore class is saved in this file
from src.logger import get_logger

logger =  get_logger(__name__)

# Initialize Flask app
app = Flask(__name__, template_folder='templates')


# Initialize the model
MODEL_PATH = 'artifacts/models/random_forest_model.pkl'
with open(MODEL_PATH, 'rb') as model_file:
    model = pickle.load(model_file)

# Define the feature names
FEATURE_NAMES = ['Age', 'Fare', 'Sex', 'Embarked', 'Familysize', 'Isalone', 'HasCabin',
                 'Title', 'Pclass_Fare', 'Age_Fare']

# Initialize RedisFeatureStore instance
feature_store = RedisFeatureStore()

# Initialize the drift detector (KSDrift for Data Drift Detection)
scaler = StandardScaler()


# Function to fit the scaler with historical data
def fit_scaler_on_reference_data():
    """Fit the StandardScaler on reference data from Redis."""
    # Retrieve all entity IDs from Redis and their features
    entity_ids = feature_store.get_all_entity_ids()
    all_features = feature_store.get_batch_features(entity_ids)

    # Convert the features of all entities into a DataFrame
    all_features_df = pd.DataFrame.from_dict(all_features, orient='index')[FEATURE_NAMES]

    # Fit the scaler on the historical data (this will compute the mean and std dev)
    scaler.fit(all_features_df)

    # Return the scaled historical data for drift detection
    return scaler.transform(all_features_df)

# Fit the scaler and drift detector with the reference data when the app starts
historical_data_scaled = fit_scaler_on_reference_data()
ksd = KSDrift(x_ref=historical_data_scaled, p_val=0.05)  # Initialize KSDrift with historical data

@app.route('/')
def home():
    """Render the HTML form for input."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle form submission and make predictions."""
    try:
        # Extract and validate form data
        data = request.form
        Age = float(data['Age'])
        Fare = float(data['Fare'])
        Sex = int(data['Sex'])
        Embarked = int(data['Embarked'])
        Familysize = int(data['Familysize'])
        Isalone = int(data['Isalone'])
        HasCabin = int(data['HasCabin'])
        Title = int(data['Title'])
        Pclass_Fare = float(data['Pclass_Fare'])
        Age_Fare = float(data['Age_Fare'])

        # Prepare feature DataFrame for prediction
        features = pd.DataFrame([[Age, Fare, Sex, Embarked, Familysize, Isalone, HasCabin,
                                  Title, Pclass_Fare, Age_Fare]], columns=FEATURE_NAMES)

        # Scale features using the already-fitted scaler
        features_scaled = scaler.transform(features)

        # Check for data drift
        drift = ksd.predict(features_scaled)

        # Print drift response to inspect the structure
        print("Drift response:", drift)

        # Check if drift exists in the response and is significant
        # Access the drift response
        drift_response = drift.get('data', {})

        # Safely access the 'is_drift' value inside the response
        is_drift = drift_response.get('is_drift', None)

        # Check if 'is_drift' exists and if its value is 1
        if is_drift is not None and is_drift == 1:
            print("Data drift detected!")
            logger.info("Data Drift Detected.")

        # Make prediction
        prediction = model.predict(features)[0]

        # Interpret prediction
        result = 'Survived' if prediction == 1 else 'Did not survive'

        # Return prediction result
        return render_template('index.html', prediction_text=f'The prediction is: {result}')

    except KeyError as e:
        return jsonify({'error': f'Missing input for field: {str(e)}'})

    except ValueError as e:
        return jsonify({'error': f'Invalid input type: {str(e)}'})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
