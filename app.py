from flask import Flask, request, jsonify
import pandas as pd
import joblib
from config import MODEL_PATH

# Load the trained model
app = Flask(__name__)
model = joblib.load(MODEL_PATH)

# Load the features used during training from the model's training data
trained_features = model.feature_names_in_  # `feature_names_in_` contains the features seen during training

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()  # Get JSON data
    df = pd.DataFrame(data)  # Convert to DataFrame

    # Check for any missing columns and fill with default values (e.g., 0)
    for feature in trained_features:
        if feature not in df.columns:
            df[feature] = 0

    # Ensure the input DataFrame has the same feature order as in training
    df = df[trained_features]

    # Make predictions
    predictions = model.predict(df)
    return jsonify(predictions.tolist())

if __name__ == '__main__':
    app.run(debug=True)
