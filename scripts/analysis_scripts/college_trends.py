import pandas as pd
import matplotlib.pyplot as plt

# load the dataset
df = pd.read_csv('data/campus_reports.csv')

# Group by College/University and count incidents
incidents_per_college = df.groupby('College/University')['Date of Incident'].count().reset_index()
incidents_per_college.columns = ['College', 'Incident Count']

# Sort by Incident Count in descending order
incidents_per_college = incidents_per_college.sort_values(by='Incident Count', ascending=False)

# Select the top 20 colleges
top_colleges = incidents_per_college.head(20)

# Plot the bar chart for the top 20 colleges
plt.figure(figsize=(10, 6))
plt.bar(top_colleges['College'], top_colleges['Incident Count'], color='navy')
plt.xticks(rotation=75, ha='right')  # Rotate college names for readability
plt.xlabel('College')
plt.ylabel('Number of Incidents')
plt.title('Top 20 Colleges with Most Incidents')

# Show the plot
plt.tight_layout()
plt.savefig('visualizations/college_trends.png', bbox_inches='tight', dpi=300)
plt.show()