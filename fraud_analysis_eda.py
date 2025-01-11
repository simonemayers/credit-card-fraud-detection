# fraud_analysis_eda.py

import matplotlib.pyplot as plt
import seaborn as sns
from utils import load_data, scale_amount

# Load and preprocess data
df = load_data()
df = scale_amount(df)

# Plot the distribution of the Class column
plt.figure(figsize=(6, 4))
sns.countplot(x='Class', data=df)
plt.title("Distribution of Fraud vs Non-Fraud Transactions")
plt.xlabel("Class (0: Non-Fraud, 1: Fraud)")
plt.ylabel("Count")
plt.show()

# Plot distribution of transaction Amounts for Fraud vs Non-Fraud
plt.figure(figsize=(10, 6))
sns.histplot(df['Amount'][df['Class'] == 0], color='blue', label='Non-Fraud', kde=True, stat="density")
sns.histplot(df['Amount'][df['Class'] == 1], color='red', label='Fraud', kde=True, stat="density")
plt.title("Transaction Amounts for Fraud vs Non-Fraud Transactions")
plt.xlabel("Transaction Amount")
plt.legend()
plt.show()

# Plot correlation matrix
plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(), cmap="coolwarm", annot=False, fmt=".2f")
plt.title("Correlation Matrix of Features")
plt.show()

# Plot distribution of selected principal components
components = ['V1', 'V2', 'V3']
plt.figure(figsize=(15, 5))
for i, comp in enumerate(components, 1):
    plt.subplot(1, 3, i)
    sns.histplot(df[comp][df['Class'] == 0], color='blue', label='Non-Fraud', kde=True, stat="density")
    sns.histplot(df[comp][df['Class'] == 1], color='red', label='Fraud', kde=True, stat="density")
    plt.title(f"Distribution of {comp}")
    plt.legend()
plt.tight_layout()
plt.show()
