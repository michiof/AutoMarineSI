import os
import pandas as pd
import openai
import csv
import time
from dotenv import load_dotenv

# Define input and output file names
input_file = './data/accident-db.csv'
output_file_csv = './data/accident-db_emb.csv'

# Load the .env.local file
load_dotenv('.env.local')

# Define the model
model = "text-embedding-ada-002"

openai.api_key = os.environ.get("OPENAI_API_KEY")

# Read the CSV file
df = pd.read_csv(input_file, encoding='utf-8')

print("Completed loading data")

print("Start embeddings calculation....")

# Remove the output file if it exists (since we're appending to it)
if os.path.exists(output_file_csv):
    os.remove(output_file_csv)

# Define batch size to avoid API rate limit
batch_size = 5000

# Calculate number of batches
num_batches = len(df) // batch_size + 1

for batch_num in range(num_batches):
    start = batch_num * batch_size
    end = start + batch_size

    batch_df = df.iloc[start:end]

    for i, row in batch_df.iterrows():
        embedding = openai.Embedding.create(
            input=row['Cause'], model=model
        )["data"][0]["embedding"]

        # Convert embedding from list to string to save in csv
        embedding_str = ', '.join(map(str, embedding))

        row['Cause_embeddings'] = embedding_str

        # Append row to the output file
        with open(output_file_csv, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if f.tell() == 0:
                # Write header if file is empty
                writer.writerow(row.index)
            writer.writerow(row)

    # Pause after each batch to avoid hitting API rate limit
    print(f"Completed embeddings calculation for batch {batch_num+1} and saved to csv file.")
    if batch_num != num_batches - 1:  # No need to sleep after last batch
        print("Sleeping for 60 seconds to avoid API rate limit...")
        time.sleep(60)
        print("Start embeddings calculation....")

print("Completed embeddings calculation for all batches and saved to csv file.")
