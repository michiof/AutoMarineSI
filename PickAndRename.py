import os
import pandas as pd
from dotenv import load_dotenv

# Define input and output file names
input_file = 'JapanTransportSafetyBoard202306download.csv'
output_file_csv = './data/accident-db.csv'

# Specify the path to the file containing the new column names
rename_file = './data/rename.txt'

# Load the .env.local file
load_dotenv('.env.local')

# Read the rename mapping into a dictionary
with open(rename_file, 'r', encoding='utf-8') as f:
    rename_dict = dict(line.strip().split() for line in f)

# Choose your desired columns here. These should be the original column names.
columns_to_pick = ['報告書番号', '発生年月日', '事故等種類', '事故等名', '発生場所', '報告書（PDF）公表', '概要', '原因']

# Ensure that your desired columns exist in the rename dictionary
if not all(column in rename_dict for column in columns_to_pick):
    print('Some columns are missing in the rename file.')
    exit(1)

# Read the CSV file
df = pd.read_csv(input_file, encoding='utf-8')

# Select the desired columns and make a copy
df_selected = df[columns_to_pick].copy()

# Rename the columns
df_selected.columns = df_selected.columns.map(rename_dict)

print("Completed loading and rename")

# Save renamed DataFrame to a CSV file
df_selected.to_csv(output_file_csv, index=False, encoding='utf-8')

print(f"Renamed data saved to {output_file_csv}.")