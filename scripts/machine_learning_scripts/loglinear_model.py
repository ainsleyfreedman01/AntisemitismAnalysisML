import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

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
X = sm.add_constant(X)  # Add a constant for the intercept (required for statsmodels)

# Log-transform the incident counts (target variable) to model exponential growth
y = np.log(incident_counts['Incident Count'])

# Fit linear regression model on log-transformed data
model = sm.OLS(y, X).fit()

# Print model summary
print(model.summary())

# Make predictions for the next 5 years
next_5_years = pd.DataFrame({'Year': [2025, 2026, 2027, 2028, 2029]})
next_5_years = sm.add_constant(next_5_years)  # Add constant for intercept

# Predict on log scale, then back-transform using exp()
log_predictions = model.predict(next_5_years)
predictions = np.exp(log_predictions)  # Exponentiate to get predictions in original scale

# Print predictions for future years
for year, prediction in zip(next_5_years['Year'], predictions):
    print(f"Predicted Incident Count for {year}: {int(round(prediction))} incidents")

# Combine original and future data for plotting the connected line
combined_years = pd.concat([incident_counts['Year'], next_5_years['Year']], ignore_index=True)
combined_X = pd.DataFrame({'Year': combined_years})
combined_X = sm.add_constant(combined_X)  # Add constant for intercept

# Predict for the combined years
combined_predictions_log = model.predict(combined_X)
combined_predictions = np.exp(combined_predictions_log)  # Exponentiate predictions to get original scale

# Plot the combined exponential growth line
plt.plot(combined_years, combined_predictions, color='red', label='Exponential Growth Line')

plt.xlabel('Year')
plt.ylabel('Incident Count')
plt.title('Prediction of Antisemtic Incident Counts on College Campuses by Year (Log-Linear Model)', fontsize=10)
plt.legend()

plt.savefig('visualizations/loglinear_model.png', bbox_inches='tight', dpi=300)
plt.show()
