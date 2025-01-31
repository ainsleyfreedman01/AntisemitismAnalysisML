import pandas as pd
import matplotlib.pyplot as plt

# load the dataset
df = pd.read_csv('data/campus_reports.csv')
# Ensure 'Date of Incident' is in datetime format
df['Date of Incident'] = pd.to_datetime(df['Date of Incident'], format='%m/%d/%y', errors='coerce')

# Extract month and year for grouping
df['Month'] = df['Date of Incident'].dt.month
df['Year'] = df['Date of Incident'].dt.year

# Group by month and year, then count incidents
monthly_trends = df.groupby(['Year', 'Month']).size().unstack(level=0)

# Plot
plt.figure(figsize=(12, 6))
monthly_trends.plot(marker='o', linestyle='-', figsize=(12, 6), cmap='tab10')

# Labels and title
plt.xlabel('Month')
plt.ylabel('Number of Incidents')
plt.title('Seasonal Trends in Incidents by Year')
plt.xticks(ticks=range(1, 13), labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.legend(title='Year', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, linestyle='--', alpha=0.6)

# Save and show
plt.tight_layout()
plt.savefig('visualizations/seasonal_trends.png', bbox_inches='tight', dpi=300)
plt.show()