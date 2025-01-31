import pandas as pd
import matplotlib.pyplot as plt

# load the dataset
df = pd.read_csv('data/campus_reports.csv')

# Group by City of Incident and count incidents
incidents_per_city = df.groupby('City of Incident')['Date of Incident'].count().reset_index()
incidents_per_city.columns = ['City', 'Incident Count']

# Sort by Incident Count in descending order
incidents_per_city = incidents_per_city.sort_values(by='Incident Count', ascending=False)

# Select the top 10 cities
top_cities = incidents_per_city.head(30)

# Plot the bar chart for the top 10 cities
plt.figure(figsize=(10, 6))
plt.bar(top_cities['City'], top_cities['Incident Count'], color='skyblue')
plt.xticks(rotation=65, ha='right')  # Rotate city names for readability
plt.xlabel('City')
plt.ylabel('Number of Incidents')
plt.title('Top 30 Cities with Most Incidents')

# Show the plot
plt.tight_layout()
plt.savefig('visualizations/city_state_trends.png', bbox_inches='tight', dpi=300)
plt.show()