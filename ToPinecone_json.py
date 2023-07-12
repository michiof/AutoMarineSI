#To upsert Pinecone DB from JSON file (not tested)

import os
import pandas as pd
import ast
import pinecone
from dotenv import load_dotenv
import json

# Load the .env.local file
load_dotenv('.env.local')

pinecone_api_key = os.environ.get("PINECONE_API_KEY")
pinecone_environment = os.environ.get("PINECONE_ENVIRONMENT")

# Define input and output file names
input_file_json = './data/accident-db_emb.json'

# Load data from the JSON file
with open(input_file_json, 'r') as file:
    data = json.load(file)

# Convert the data to a DataFrame
df = pd.DataFrame(data)

# Handle NaN values in the DataFrame
df = df.fillna('Unknown')

# Initialize Pinecone
pinecone.init(api_key=pinecone_api_key, environment=pinecone_environment)

# Create the new Index if not exists
index_name = "accident-db"
if index_name not in pinecone.list_indexes():
    print("Start creating the new index in Pinecone. It takes some time.")
    pinecone.create_index(index_name, dimension=1536, metric="cosine", pod_type="p1")
    print("Created the new index in Pinecone")

# Connecting to the index
index = pinecone.Index(index_name)

print("Start upsert to Pinecone database")

# Upsert vectors into the index
for i, row in df.iterrows():
    # Convert string representation of embedding back to list
    embeddings = list(map(float, row["Cause_embeddings"].split(', ')))

    vector = (
        f"id_{i}", 
        embeddings, 
        {
            "ReportID": row["ReportID"], 
            "Date_accident": row["Date_accident"], 
            "Type_accident": row["Type_accident"], 
            "Title": row["Title"], 
            "Place": row["Place"], 
            "URL": row["URL"], 
            "Outline": row["Outline"], 
            "Cause": row["Cause"]
        }
    )
    index.upsert([vector])

print("Completed upsert to Pinecone database")