# Credit Card Fraud Detection

## Project Overview
This project aims to detect fraudulent transactions in a credit card dataset using machine learning models. It includes data preprocessing, exploratory data analysis (EDA), model training, and deployment as a REST API. The primary model used for fraud detection is a **Random Forest Classifier**, chosen for its interpretability and robustness.

## Project Structure
The project is organized into modular files for easy maintenance and scalability:

```
credit-card-fraud-detection/
│
├── app.py                # Flask API setup for deployment
├── config.py             # Configuration file for parameters and file paths
├── fraud_analysis.py     # Main script for data loading, preprocessing, and model training/evaluation
├── fraud_analysis_eda.py # EDA script for visualizing the dataset
├── model_training.py     # Model training and evaluation functions
├── utils.py              # Helper functions for data loading and preprocessing
└── creditcard_2023.csv   # Dataset file
```

## Getting Started

### Prerequisites
- **Python 3.8+** 
- **Required Packages**: Install the necessary Python packages using `pip`:
  
  ```bash
  pip install -r requirements.txt
  ```

  Ensure you have the following libraries installed:
  - `pandas`
  - `scikit-learn`
  - `matplotlib`
  - `seaborn`
  - `xgboost`
  - `flask`
  - `joblib`

### Dataset
The dataset is stored in `creditcard_2023.csv`. It contains anonymized credit card transactions labeled as fraud (1) or non-fraud (0).

### Configuration
The `config.py` file includes:
- **Paths** to the dataset and saved model.
- **Model parameters** for the Random Forest and XGBoost models.

You can adjust these settings as needed.

## Running the Project

### 1. Exploratory Data Analysis (EDA)
To perform EDA, run `fraud_analysis_eda.py`:

```bash
python fraud_analysis_eda.py
```

This script will generate visualizations, including:
- Distribution of fraud vs. non-fraud transactions.
- Distribution of transaction amounts.
- Correlation matrix.
- Principal component analysis for feature relationships.

### 2. Model Training and Evaluation
To train and evaluate the model, run `fraud_analysis.py`:

```bash
python fraud_analysis.py
```

This script will:
- Load and preprocess the data.
- Split data into training and testing sets.
- Train multiple models (Logistic Regression, Random Forest, and XGBoost).
- Evaluate the models and print classification reports and ROC AUC scores.
- Save the trained Random Forest model to `random_forest_model.joblib`.

### 3. Deploying the Model as a REST API
To set up the model as a REST API, run `app.py`:

```bash
python app.py
```

This will start a Flask API on `http://127.0.0.1:5000` with a `/predict` endpoint. You can make POST requests to this endpoint to get predictions.

#### Example Request
Send a JSON payload with transaction data to the `/predict` endpoint using a tool like **curl** or **Postman**:

```bash
curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d '[{"V1": 0.1, "V2": -0.2, "V3": 0.3, "Amount": 1000, ...}]'
```

## Project Files

- **app.py**: Defines the Flask API and prediction endpoint.
- **config.py**: Stores configuration details like file paths and model parameters.
- **fraud_analysis.py**: The main script for data loading, preprocessing, and training.
- **fraud_analysis_eda.py**: Generates visualizations for data exploration.
- **model_training.py**: Contains functions for model training, evaluation, and saving.
- **utils.py**: Helper functions for loading data and scaling features.

## Future Improvements
- **Hyperparameter Tuning**: Fine-tune model parameters to improve accuracy.
- **Additional Models**: Experiment with other machine learning models.
- **Automated Model Retraining**: Set up periodic model retraining with new data.
- **API Documentation**: Use tools like Swagger for documenting the API.

## Acknowledgments
- The dataset for this project was sourced from Kaggle.
- Inspired by the need to protect financial institutions and customers from fraudulent transactions.

