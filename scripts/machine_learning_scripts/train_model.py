import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

# Load the dataset
df = pd.read_csv('data/campus_reports.csv')

# Preprocess the data
df['Date of Incident'] = pd.to_datetime(df['Date of Incident'], format='%m/%d/%y', errors='coerce')

# Extract the year from the 'Date of Incident' column
df['Year'] = df['Date of Incident'].dt.year

# Aggregate the data by year and count incidents
incident_counts = df.groupby('Year').size().reset_index(name='Incident Count')

# Print aggregated data for verification
print("Incident Counts by Year:\n", incident_counts)

# Create the feature (X) and target (y)
X = incident_counts[['Year']]  # Year is the feature
y = incident_counts['Incident Count']  # Incident count is the target

# Train a linear regression model
model = LinearRegression()
model.fit(X, y)

# Make a prediction for next 5 years
next_5_years = pd.DataFrame({'Year': [2025, 2026, 2027, 2028, 2029]})
predictions = model.predict(next_5_years)

# Print predictions
for year, prediction in zip(next_5_years['Year'], predictions):
    print(f"Incident Count for {year}: {int(prediction)} incidents")
