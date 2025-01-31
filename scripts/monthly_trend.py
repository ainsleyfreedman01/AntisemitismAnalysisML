import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# load the dataset
df = pd.read_csv('data/campus_reports.csv')

# Convert 'Date of Incident' column to datetime format
df['Date of Incident'] = pd.to_datetime(df['Date of Incident'], format='%m/%d/%y', errors='coerce')

# Group by 'Date of Incident' and count incidents
df['Month'] = df['Date of Incident'].dt.to_period('M')
monthly_trend = df.groupby('Month').size()
monthly_trend.index = monthly_trend.index.to_timestamp()

plt.figure(figsize=(10, 6))
plt.plot(monthly_trend.index, monthly_trend.values)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%Y'))
plt.xlabel('Month')
plt.ylabel('Number of Incidents')
plt.title('Monthly Trend of Antisemitic Incidents on College Campuses')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('visualizations/monthly_trend.png', bbox_inches='tight', dpi=300)
plt.show()