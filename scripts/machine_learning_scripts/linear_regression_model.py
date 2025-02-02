import pandas as pd
from sklearn.linear_model import LinearRegression
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

# Plot the linear regression line through all data (historical and predicted)
full_years = pd.concat([incident_counts['Year'], next_5_years['Year']])  # Combine years for both historical and predicted data
full_predictions = model.predict(full_years.values.reshape(-1, 1))  # Predict using the linear model

# Plot the combined linear regression line
plt.plot(full_years, full_predictions, color='lightgreen', label='Linear Regression Line')

# Plot the original data points (dots)
plt.plot(incident_counts['Year'], incident_counts['Incident Count'], 'o', label='Incident Counts')


# Plot the predicted future data points
plt.plot(next_5_years['Year'], predictions, 'ro', label='Predicted Incident Counts')

# Labels and title
plt.xlabel('Year')
plt.ylabel('Incident Count')
plt.title('Prediction of Antisemtic Incident Counts on College Campuses by Year (Linear Regression Model)', fontsize=10)
plt.legend()
plt.show()

plt.savefig('visualizations/linear_regression_model.png', bbox_inches='tight', dpi=300)