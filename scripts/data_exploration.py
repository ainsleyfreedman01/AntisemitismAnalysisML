import pandas as pd
import matplotlib.pyplot as plt

# load the dataset
df = pd.read_csv('data/campus_reports.csv')

print("\nFirst 5 rows:")
print(df.head())

print("\nSummary statistics:")
print(df.describe())

print("\nColumn Uniqueness:")
print(df['City of Incident'].unique())
print(df['State of Incident'].unique())
print(df['Incident Type'].unique())

print(df['Date of Incident'].head())

df['Date of Incident'] = pd.to_datetime(df['Date of Incident'], format='%m/%d/%y', errors='coerce')
print(df[df['Date of Incident'].isna()])
df['Year'] = df['Date of Incident'].dt.year

print(df.columns)
print(df.head())

incidents_per_year = df.groupby('Year')['Date of Incident'].count()
print("\nIncidents per year:")
print(incidents_per_year)

# Create the bar plot
ax = incidents_per_year.plot(kind='bar', color='skyblue')

# Add data labels above each bar
for i in range(len(incidents_per_year)):
    plt.text(i, incidents_per_year.iloc[i] + 10.0, str(incidents_per_year.iloc[i]), ha='center', fontsize=8)

# Labeling the plot
plt.xlabel('Year')
plt.ylabel('Number of Incidents')
plt.title('Incidents per Year')
# plt.savefig('incidents_per_year.png' , bbox_inches='tight' , dpi=300)
# Show the plot
plt.show()