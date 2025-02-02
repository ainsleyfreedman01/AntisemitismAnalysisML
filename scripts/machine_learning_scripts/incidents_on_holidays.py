import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# Load the dataset
df = pd.read_csv('data/campus_reports.csv')

df['Date of Incident'] = pd.to_datetime(df['Date of Incident'], format='%m/%d/%y', errors='coerce')

jewish_holidays = {
    '3/6/2023': 'Purim',
    '4/5/2023': 'Passover',
    '4/6/2023': 'Passover',
    '4/7/2023': 'Passover',
    '4/8/2023': 'Passover',
    '4/9/2023': 'Passover',
    '4/10/2023': 'Passover',
    '4/11/2023': 'Passover',
    '4/12/2023': 'Passover',
    '4/13/2023': 'Passover',
    '4/18/2023': 'Yom HaShoah',
    '4/26/2023': "Yom Ha'atzmaut",
    '9/15/2023': 'Rosh Hashana',
    '9/16/2023': 'Rosh Hashana',
    '9/17/2023': 'Rosh Hashana',
    '9/24/2023': 'Yom Kippur',
    '9/25/2023': 'Yom Kippur',
    '9/29/2023': 'Sukkot',
    '9/30/2023': 'Sukkot',
    '10/1/2023': 'Sukkot',
    '10/2/2023': 'Sukkot',
    '10/3/2023': 'Sukkot',
    '10/4/2023': 'Sukkot',
    '10/5/2023': 'Sukkot',
    '10/6/2023': 'Sukkot',
    '10/7/2023': 'Simchat Torah',
    '10/8/2023': 'Simchat Torah',
    '12/7/2023': 'Chanukah',
    '12/8/2023': 'Chanukah',
    '12/9/2023': 'Chanukah',
    '12/10/2023': 'Chanukah',
    '12/11/2023': 'Chanukah',
    '12/12/2023': 'Chanukah',
    '12/13/2023': 'Chanukah',
    '12/14/2023': 'Chanukah',
    '12/15/2023': 'Chanukah',
    
    '3/23/2024': 'Purim',
    '3/24/2024': 'Purim',
    '4/22/2024': 'Passover',
    '4/23/2024': 'Passover',
    '4/24/2024': 'Passover',
    '4/25/2024': 'Passover',
    '4/26/2024': 'Passover',
    '4/27/2024': 'Passover',
    '4/28/2024': 'Passover',
    '4/29/2024': 'Passover',
    '4/30/2024': 'Passover',
    '10/2/2024': 'Rosh Hashana',
    '10/3/2024': 'Rosh Hashana',
    '10/4/2024': 'Rosh Hashana',
    '10/11/2024': 'Yom Kippur',
    '10/12/2024': 'Yom Kippur',
    '10/16/2024': 'Sukkot',
    '10/17/2024': 'Sukkot',
    '10/18/2024': 'Sukkot',
    '10/19/2024': 'Sukkot',
    '10/20/2024': 'Sukkot',
    '10/21/2024': 'Sukkot',
    '10/22/2024': 'Sukkot',
    '10/23/2024': 'Simchat Torah',
    '10/24/2024': 'Simchat Torah',
    '12/25/2024': 'Chanukah',
    '12/26/2024': 'Chanukah',
    '12/27/2024': 'Chanukah',
    '12/28/2024': 'Chanukah',
    '12/29/2024': 'Chanukah',
    '12/30/2024': 'Chanukah',
    '12/31/2024': 'Chanukah'
}

national_holidays = {
    '1/1/2023': "New Year's Day",
    '1/16/2023': 'Martin Luther King Jr. Day',
    '2/20/2023': "Presidents' Day",
    '4/7/2023': 'Good Friday',
    '4/9/2023': 'Easter Sunday',
    '5/29/2023': 'Memorial Day',
    '7/4/2023': 'Independence Day',
    '9/4/2023': 'Labor Day',
    '10/31/2023': 'Halloween',
    '11/7/2023': 'Election Day',
    '11/11/2023': 'Veterans Day',
    '11/23/2023': 'Thanksgiving Day',
    '12/24/2023': 'Christmas Eve',
    '12/25/2023': 'Christmas Day',
    '12/31/2023': "New Year's Eve",
    
    '1/1/2024': "New Year's Day",
    '1/15/2024': 'Martin Luther King Jr. Day',
    '2/19/2024': "Presidents' Day",
    '3/29/2024': 'Good Friday',
    '3/31/2024': 'Easter Sunday',
    '5/27/2024': 'Memorial Day',
    '7/4/2024': 'Independence Day',
    '9/2/2024': 'Labor Day',
    '10/31/2024': 'Halloween',
    '11/5/2024': 'Election Day',
    '11/11/2024': 'Veterans Day',
    '11/28/2024': 'Thanksgiving Day',
    '12/24/2024': 'Christmas Eve',
    '12/25/2024': 'Christmas Day',
    '12/31/2024': "New Year's Eve"
}

october_7th_attacks = {
    '10/7/2023': 'October 7th Attacks',
    '10/7/2024': '1 year anniversary'
}

# Ensure that 'Date of Incident' is a datetime object
df['Date of Incident'] = pd.to_datetime(df['Date of Incident'], errors='coerce')

# Function to match the date with a holiday

# Count incidents per date and sort by most incidents
incident_counts = df.groupby('Date of Incident').size().reset_index(name='Incident Count')
incident_counts = incident_counts.sort_values(by='Incident Count', ascending=False)

# Function to check if a date matches a holiday
def get_holiday_name(date):
    date_str = date.strftime('%-m/%-d/%Y')  # Format date to match dictionary keys
    return (
        jewish_holidays.get(date_str) or 
        national_holidays.get(date_str) or 
        october_7th_attacks.get(date_str) or 
        None
    )

# Apply function to match holidays
incident_counts['Holiday'] = incident_counts['Date of Incident'].apply(get_holiday_name)

# Filter to only include dates that match a holiday
holiday_incidents = incident_counts.dropna(subset=['Holiday'])
jewish_holiday_incidents = incident_counts[incident_counts['Holiday'].isin(jewish_holidays.values())]
national_holiday_incidents = incident_counts[incident_counts['Holiday'].isin(national_holidays.values())]

# Display the top holidays with the most incidents
print("\nTop holidays with the most incidents:\n", holiday_incidents.head(20))

# Display the top Jewish holidays with the most incidents
print("\nTop Jewish holidays with the most incidents:\n", jewish_holiday_incidents.head(20))

# Display the top National holidays with the most incidents
print("\nTop National holidays with the most incidents:\n", national_holiday_incidents.head(10))
