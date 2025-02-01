import pandas as pd

def preprocess_data(file_path):
    df = pd.read_csv('data/campus_reports.csv')
    print(df.head())

    print(df.isnull().sum()) # this is zero, so we're good!

    df['Date of Incident'] = pd.to_datetime(df['Date of Incident'], format='%m/%d/%y', errors='coerce')

    print(df[df['Date of Incident'].isna()]) # this is empty, so we're good here, too!

    df['Year'] = df['Date of Incident'].dt.year

    print(df.head())
    return df

processed_data = preprocess_data('data/campus_reports.csv')
print("The processed data is:\n",processed_data.head())