import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# load the dataset
df = pd.read_csv('data/campus_reports.csv')

# Convert 'Date of Incident' column to datetime format
df['Date of Incident'] = pd.to_datetime(df['Date of Incident'], format='%m/%d/%y', errors='coerce')

# Group by 'Date of Incident' and count incidents
incidents_over_time = df.groupby('Date of Incident').size()

plt.figure(figsize=(10, 6))
plt.plot(incidents_over_time.index, incidents_over_time.values)

highlight_date = pd.Timestamp('2023-10-07')
plt.axvline(x=highlight_date, color='red', linestyle='--', linewidth=2, label='October 7th Attack')

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())

plt.xlabel('Date')
plt.ylabel('Number of Incidents')
plt.title('Antisemitism Trends on College Campuses')
plt.xticks(rotation=45)
plt.tight_layout()
plt.legend()
plt.savefig('visualizations/incidents_over_time.png', bbox_inches='tight', dpi=300)
plt.show()