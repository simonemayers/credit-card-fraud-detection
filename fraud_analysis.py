# fraud_analysis.py

# Import necessary functions from our modules
from model_training import train_and_evaluate_model
from utils import load_data, scale_amount
from sklearn.model_selection import train_test_split

# Load and preprocess data
df = load_data()
df = scale_amount(df)

# Split data into features (X) and target (y)
X = df.drop(columns=['Class'])
y = df['Class']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y
)

# Train and evaluate the model
train_and_evaluate_model(X_train, y_train, X_test, y_test)
