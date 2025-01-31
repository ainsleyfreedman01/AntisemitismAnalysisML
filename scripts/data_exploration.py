import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import matplotlib.dates as mdates

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
#plt.show()

######################################################
# Dictionary for state abbreviations
state_abbreviations = {
    'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR', 'California': 'CA',
    'Colorado': 'CO', 'Connecticut': 'CT', 'Delaware': 'DE', 'Florida': 'FL', 'Georgia': 'GA',
    'Hawaii': 'HI', 'Idaho': 'ID', 'Illinois': 'IL', 'Indiana': 'IN', 'Iowa': 'IA',
    'Kansas': 'KS', 'Kentucky': 'KY', 'Louisiana': 'LA', 'Maine': 'ME', 'Maryland': 'MD',
    'Massachusetts': 'MA', 'Michigan': 'MI', 'Minnesota': 'MN', 'Mississippi': 'MS',
    'Missouri': 'MO', 'Montana': 'MT', 'Nebraska': 'NE', 'Nevada': 'NV', 'New Hampshire': 'NH',
    'New Jersey': 'NJ', 'New Mexico': 'NM', 'New York': 'NY', 'North Carolina': 'NC',
    'North Dakota': 'ND', 'Ohio': 'OH', 'Oklahoma': 'OK', 'Oregon': 'OR', 'Pennsylvania': 'PA',
    'Rhode Island': 'RI', 'South Carolina': 'SC', 'South Dakota': 'SD', 'Tennessee': 'TN',
    'Texas': 'TX', 'Utah': 'UT', 'Vermont': 'VT', 'Virginia': 'VA', 'Washington': 'WA',
    'West Virginia': 'WV', 'Wisconsin': 'WI', 'Wyoming': 'WY'
}

# Convert state names to abbreviations
df['State of Incident'] = df['State of Incident'].map(lambda x: state_abbreviations.get(x, x))

incidents_per_state = df.groupby('State of Incident')['Date of Incident'].count().reset_index()
incidents_per_state.columns = ['State', 'Incident Count']

fig = px.choropleth(incidents_per_state, locations='State', locationmode='USA-states', color='Incident Count',
                    scope='usa', color_continuous_scale='Reds', title='Antisemitic Incidents per State')
# fig.show()
# fig.write_html('incidents_per_state.html')

######################################################

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
plt.savefig('incidents_over_time.png', bbox_inches='tight', dpi=300)
plt.show()