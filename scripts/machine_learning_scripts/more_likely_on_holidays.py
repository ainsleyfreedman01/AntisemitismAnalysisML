import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Sample DataFrame
df = pd.read_csv("data/campus_reports.csv")  # Load your dataset

# Calculate average incidents on non-holidays
average_incidents = df[df["is_holiday"] == 0]["incident_count"].mean()

# Create a binary target column
df["is_high_incident_day"] = (df["incident_count"] > average_incidents).astype(int)

# Select Features
X = df[["is_holiday", "day_of_week", "holiday_type"]]  # Add more features if available
y = df["is_high_incident_day"]

# Encode categorical variables (if needed)
X = pd.get_dummies(X, drop_first=True)

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make Predictions
y_pred = model.predict(X_test)

# Evaluate Performance
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
