import openai  # for calling the OpenAI API
import pinecone
import os
import pandas as pd  # for storing text and embeddings data
import tiktoken  # for counting tokens
from scipy import spatial  # for calculating vector similarities for search
from dotenv import load_dotenv  # import the load_dotenv function

# To show a "...waiting..." message while waiting for a response from the API
import threading
import time
import sys

# Specify the output Language
LANGUAGE = "Japanese"

# Models
EMBEDDING_MODEL = "text-embedding-ada-002"
GPT_MODEL = "gpt-3.5-turbo"

# Define the Pinecone index
INDEX_NAME = "accident-db"

# Load environment variables from .env.local file
load_dotenv('.env.local')  # replace with the path to your .env.local file if it's not in the same directory

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
    top_n: int = 50
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


def query_message(
    query: str,
    model: str,
    token_budget: int
) -> str:
    """Return a message for GPT, with relevant source texts pulled from a dataframe."""
    results = strings_ranked_by_relatedness(query)

    introduction = '''
    Create a potential accident that could occur in the future based on the given nearmiss report. 
    The procedure is as follows:
    First, sumarize inputed past accidents. If it was not possible, respond with "I cannot advise you...".
    Start with "Foreseeable accident summary: " Then, output the samarized sentences as you determine the accident you foresee based on past accident cases.
    The output should be wrote with 1000 characters.
    Next, produce its safety countermeasures. Start with "Safety measures: "
    '''

    language = f"\nWrite all your output in {LANGUAGE}"

    question = f"\n\nNearmiss report: {query}"

    message = introduction + language + question
    for i, match in enumerate(results["matches"], start=1):
        title = match["metadata"]["Title"]
        outline = match["metadata"]["Outline"]
        cause = match["metadata"]["Cause"]
        next_article = f'\n\n\nPast accident {i}:\n\n\nTitle:\n{title}\n\n\nOutline:\n{outline}\n\n\nCause:\n{cause}\n\n\n'
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
    token_budget: int = 4096 - 500,
    print_message: bool = False,
) -> str:
    """Answers a query using GPT and a dataframe of relevant texts and embeddings."""
    message = query_message(query, model=model, token_budget=token_budget)
    if print_message:
        print(message)
    messages = [
        {"role": "system", "content": "You are an exceptional Captain, with extensive experience at sea. You possess the ability to foresee accidents that may potentially occur in the future."},
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
    return response_message


# get input from the user
print("\n")
print("Start AutoMarineSI...")
print("\n")
message = input("Please input your near-miss report: ")
print("\n")
print(ask(message))

# Get related strings
results = strings_ranked_by_relatedness(message, top_n=5)

# Initialize a counter for print results
counter = 1

print("\n\nThese are refereced data:")

# Print the results
for match in results["matches"]:
    print("\n")
    print("Referenced Data " + str(counter))
    print(match["metadata"]["Title"])
    print(match["metadata"]["Date_accident"])
#    print(match["metadata"]["ReportID"])
#    print(match["metadata"]["Outline"].replace(" ", ""))
    print("URL: " + match["metadata"]["URL"])
    counter += 1