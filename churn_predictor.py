import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ----- Step 1: Load dataset ----- #
input_csv = "customers.csv"
try:
    df = pd.read_csv(input_csv)
    print(f"Loaded {len(df)} rows from {input_csv}")
except FileNotFoundError:
    print(f"File {input_csv} not found. Please create the CSV first.")
    exit()

# ----- Step 2: Preprocess data ----- #
# Encode target variable (churn) to 0/1
label_encoder = LabelEncoder()
df['churn_encoded'] = label_encoder.fit_transform(df['churn'])  # Yes=1, No=0

# Features and target
feature_columns = ['age', 'tenure', 'monthly_charges', 'total_charges']
X = df[feature_columns]
y = df['churn_encoded']

# Split into training and testing (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ----- Step 3: Train model ----- #
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
print("Model training completed.")

# ----- Step 4: Evaluate model ----- #
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.2f}")

# ----- Step 5: Predict churn for all customers ----- #
df['predicted_churn'] = model.predict(X)
df['predicted_churn_label'] = df['predicted_churn'].apply(lambda x: "Yes" if x == 1 else "No")

# ----- Step 6: Save results to JSON ----- #
output_json = "churn_predictions.json"
results = df[['customer_id', 'predicted_churn_label']].to_dict(orient='records')

with open(output_json, 'w') as f:
    json.dump(results, f, indent=4)

print(f"Predicted churn results saved to {output_json}")
