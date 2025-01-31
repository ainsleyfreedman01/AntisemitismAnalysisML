import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# load the dataset
df = pd.read_csv('data/campus_reports.csv')

# Convert 'Date of Incident' column to datetime format
df['Date of Incident'] = pd.to_datetime(df['Date of Incident'], format='%m/%d/%y', errors='coerce')

# Group by 'Date of Incident' and count incidents
incidents_over_time = df.groupby('Date of Incident').size()
incidents_over_time_7day = incidents_over_time.rolling(window=7, min_periods=1).mean()
incidents_over_time_30day = incidents_over_time.rolling(window=30, min_periods=1).mean()

plt.figure(figsize=(10, 6))
plt.plot(incidents_over_time.index, incidents_over_time.values, label='Daily Incidents', color='lightblue', linestyle='dotted', alpha=0.8)
plt.plot(incidents_over_time_7day.index, incidents_over_time_7day.values, label='7-Day Moving Average', color='blue', alpha=0.8)
plt.plot(incidents_over_time_30day.index, incidents_over_time_30day.values, label='30-Day Moving Average', color='indigo', alpha=0.8)

plt.xlabel('Date')
plt.ylabel('Number of Incidents')
plt.title('Daily Incidents with 7-Day and 30-Day Rolling Averages')
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
plt.xticks(rotation=45)

plt.legend()
plt.tight_layout()
plt.savefig('visualizations/incidents_over_time_7day_30day.png', bbox_inches='tight', dpi=300)
plt.show()