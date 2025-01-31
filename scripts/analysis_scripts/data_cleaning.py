import pandas as pd

# load the dataset
df = pd.read_csv('data/campus_reports.csv')

# check for missing values
print("\nMissing values:")
print(df.isnull().sum())

print("\nRows with missing values in 'College/University' column:")
print(df[df['College/University'].isna()])

# # Get the unique values and sort them alphabetically
# unique_values_sorted = sorted(df['College/University'].unique())

# # Print the sorted unique values
# print("\nUnique values in 'College/University' after cleaning (alphabetized):")
# print(unique_values_sorted)