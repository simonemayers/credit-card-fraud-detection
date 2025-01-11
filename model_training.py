# model_training.py

import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score
import joblib
from config import DATA_PATH, MODEL_PATH, RANDOM_FOREST_PARAMS, XGBOOST_PARAMS

def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df.drop(columns=['id'])
    return df

def preprocess_data(df):
    # Standardize 'Amount' column
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    df['Amount'] = scaler.fit_transform(df[['Amount']])
    return df

def train_and_evaluate_model(X_train, y_train, X_test, y_test):
    # Train models with predefined parameters
    random_forest = RandomForestClassifier(**RANDOM_FOREST_PARAMS)
    random_forest.fit(X_train, y_train)
    
    # Evaluate
    rf_preds = random_forest.predict(X_test)
    print("Random Forest Report")
    print(classification_report(y_test, rf_preds))
    print("ROC AUC:", roc_auc_score(y_test, random_forest.predict_proba(X_test)[:, 1]))

    # Save the model
    joblib.dump(random_forest, MODEL_PATH)

if __name__ == "__main__":
    # Load and preprocess data
    df = load_data()
    df = preprocess_data(df)
    
    # Split data
    X = df.drop(columns=['Class'])
    y = df['Class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Train and evaluate
    train_and_evaluate_model(X_train, y_train, X_test, y_test)
