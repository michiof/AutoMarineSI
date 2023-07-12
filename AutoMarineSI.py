import openai  # for calling the OpenAI API
import pinecone
import os
import pandas as pd  # for storing text and embeddings data
import tiktoken  # for counting tokens
from scipy import spatial  # for calculating vector similarities for search
from dotenv import load_dotenv  # import the load_dotenv function
import json
from datetime import datetime

# To show a "...waiting..." message while waiting for a response from the API
import threading
import time
import sys


# Define the file path to config.json file
CONFIG_FILE_PATH = 'config.json'

# Define the Pinecone index
INDEX_NAME = "accident-db"

# replace with the path to your .env.local file if it's not in the same directory
load_dotenv('.env.local')

# Models
EMBEDDING_MODEL = "text-embedding-ada-002"
GPT_MODEL = "gpt-3.5-turbo"

# Token Limit
LIMIT = 4096

# Record output results in log file (Yes=1, No=0)
log_function = 1

# Set up KEYS: this now gets the API key from the .env.local file
openai.api_key = os.getenv("OPENAI_API_KEY")
pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("PINECONE_ENVIRONMENT"))

# create an instance of the index
index = pinecone.Index(INDEX_NAME)

# waiting message
def print_waiting_message(stop_event):
    print("Thinking ...", end="")
    sys.stdout.flush()  # make sure the message is immediately printed
    while not stop_event.is_set():
        print("...", end="")
        sys.stdout.flush()
        time.sleep(1)


# search function using Pinecone
def strings_ranked_by_relatedness(
    query: str,
    top_n: int = 100
) -> object:
    """Returns a list of strings and relatednesses, sorted from most related to least."""
    query_embedding_response = openai.Embedding.create(
        model=EMBEDDING_MODEL,
        input=query,
    )
    query_embedding = query_embedding_response["data"][0]["embedding"]

    results = index.query(query_embedding, top_k=top_n, include_metadata=True)
    
    return results

def num_tokens(text: str, model: str = GPT_MODEL) -> int:
    """Return the number of tokens in a string."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))


def load_from_config(file_path, key):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data[key]


def query_message(
    query: str,
    model: str,
    token_budget: int
) -> str:
    """Return a message for GPT, with relevant source texts pulled from a dataframe."""
    results = strings_ranked_by_relatedness(query)

    introduction = load_from_config(CONFIG_FILE_PATH, 'introduction')
    LANGUAGE = load_from_config(CONFIG_FILE_PATH, 'language')

    language = f"\nWrite all your output in {LANGUAGE}"
    question = f"\n\nNearmiss report: {query}"

    message = introduction + language + question
    for i, match in enumerate(results["matches"], start=1):
        title = match["metadata"]["Title"]
        outline = match["metadata"]["Outline"]
        cause = match["metadata"]["Cause"]
        url = match["metadata"]["URL"]
        next_article = f'\n\n\nPast accident {i}:\n\n\nTitle:\n{title}\n\n\nOutline:\n{outline}\n\n\nCause:\n{cause}\n\n\nURL:\n{url}\n\n\n'
        if (
            num_tokens(message + next_article + question, model=model)
            > token_budget
        ):
            break
        else:
            message += next_article

    return message

def ask(
    query: str,
    model: str = GPT_MODEL,
    token_budget: int = LIMIT - 500,
    print_message: bool = False,
) -> str:
    """Answers a query using GPT and a dataframe of relevant texts and embeddings."""
    message = query_message(query, model=model, token_budget=token_budget)
    
    system_message = load_from_config(CONFIG_FILE_PATH, 'system')

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": message},
    ]

    # Start the waiting message thread before the API call
    stop_event = threading.Event()
    waiting_thread = threading.Thread(target=print_waiting_message, args=(stop_event,))
    waiting_thread.start()

    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0.2,
    )

    # Stop the waiting message thread after the API call is completed
    stop_event.set()
    waiting_thread.join()

    print("\n")

    response_message = response["choices"][0]["message"]["content"]

    # log the results to a JSON file
    if log_function == 1:

        final_token_count = num_tokens(message, model=model)

        with open('./data/gpt_logs.json', 'a') as f:
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'token': final_token_count,
                'results': response_message,
                'prompt': message
            }
            json.dump(log_entry, f, ensure_ascii=False)
            f.write('\n')

    return response_message


# get input from the user
print("\n")
print("Start AutoMarineSI...")
print("\n")
#message = input("Please input your near-miss report: ")
message = "潮流が強く舵が効かない"
print("\n")
print(ask(message))

# Get related strings
#results = strings_ranked_by_relatedness(message, top_n=5)

# Initialize a counter for print results
#counter = 1

#print("\n\nThese are refereced data:")

# Print the results
"""
for match in results["matches"]:
    print("\n")
    print("Referenced Data " + str(counter))
    print(match["metadata"]["Title"])
    print(match["metadata"]["Date_accident"])
#    print(match["metadata"]["ReportID"])
#    print(match["metadata"]["Outline"].replace(" ", ""))
    print("URL: " + match["metadata"]["URL"])
    counter += 1
"""