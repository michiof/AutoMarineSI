# AutoMarineSI : A Near-Miss Analyzer using GPT 3 & 4

AutoMarineSI is a Python script that utilizes OpenAI's GPT3/4 to infer possible future accidents from inputted near-miss reports.

The script identifies similar past accident cases based on the content of a user-inputted near-miss report and, by leveraging GPT3/4, uses these cases to predict possible accidents. It then outputs these potential accidents along with corresponding safety measures.


## Pre-Setup

1. Install the required packages.
2. Create a .env file and specify your own keys and Pinecone environment. The format should be:  
`OPENAI_API_KEY=...`  
`PINECONE_API_KEY=...`  
`PINECONE_ENVIRONMENT=...`  
3. Specify the output language in "AutoMarinSI.py". The default is set to "Japanese".
4. Specify the OpenAI model in "AutoMarinSI.py". The default models are “gpt3.5-turbo” and “text-embedding-ada-002”.
5. Specify the OpenAI model in "AddEmbedding.py". The default model is “text-embedding-ada-002”.
6. Save your accident database in CSV format in the "data" directory. For the correct format and definitions of the accident data, refer to the sections below: ["Format and Definition of Accident Data"](https://github.com/michiof/AutoMarineSI#format-and-definition-of-accident-data) and ["Sample Data"](https://github.com/michiof/AutoMarineSI#sample-data).


## Execution

1. Run "Cal_embedding.py" to add embedding values to your CSV file. (This is required only the first time.)
2. Run "ToPinecone.py" to import the accident database into Pinecone[^1]. (This is also only required the first time.)
3. Execute "AutoMarineSI.py".
4. When you see the prompt "Please input your near-miss report:", proceed to input your report.

[^1]:When attempting to upsert to the Pinecone database, you might occasionally encounter an error message from the Pinecone server, particularly just after creating a new Pinecone index. If this occurs, don't worry. Typically, simply retrying the script resolves the issue. The second or third attempt is usually successful.



## Format and Definition of Accident Data

The "Cal_embedding.py" requires a CSV file with historical accident data. This file should include the following columns:

    1. "ReportID": A unique ID for each accident
    2. "Date_accident": The date of the accident
    3. "Type_accident": The type of the accident
    4. "Title": The title of the accident report or name of the accident
    5. "Place": The location of the accident
    6. "URL": The download URL for the accident report
    7. "Outline": An outline of the accident
    8. "Cause": The cause of the accident

Please refer to the "sample_of_accident-db.csv" file in the "data" folder for an example of the required format.


## Sample Data

### How to Use

If you don't have a past accident database, you can use the sample file "JTSB202306.csv" available in this repository. This file was downloaded from the **[Japan Transport Safety Board](https://www.mlit.go.jp/jtsb/index.html)** in June 2023. Please ensure to read the ["Copyright"](https://github.com/michiof/AutoMarineSI#copyright) below before using it.

In the file, you'll find the headers are written in Japanese. You'll need to rename these headers and remove some columns from the original file. 

The script "PickAndRename.py" is designed to perform these necessary actions. Once you've run the script, please proceed to step 1 of the [Execution](https://github.com/michiof/AutoMarineSI#execution) section: "Run Cal_embedding.py".


### Copyright

The sample file "JTSB202306.csv" in this repository was obtained from the **[Japan Transport Safety Board](https://www.mlit.go.jp/jtsb/english.html)** in June 2023. You can find the terms of use at **[https://www.mlit.go.jp/jtsb/cyo.html](https://www.mlit.go.jp/jtsb/cyo.html)**. While you are permitted to use this data, you must explicitly state that it was sourced from the **[Japan Transport Safety Board](https://www.mlit.go.jp/jtsb/index.html)** whenever you use it. Additionally, if you modify the data, you must clearly indicate that edits have been made in addition to citing the original source.



## My messages

- I've written a post that provides an overview of AutoMarineSI. Please take a look if you have any: **[English](https://www.fmcho.com/posts/2023-06-25-2)** | **[日本語](https://www.fmcho.com/posts/2023-06-25-1)**
- Although my expertise is in the Maritime industry, my proficiency in GitHub, Python, or other computer science domains is not as strong. If you are a software engineer, your assistance would be greatly appreciated.
- If you encounter any bugs or have suggestions to improve the system, I encourage you to make a pull request. All contributions are welcome.
- At present, the system has been tested solely with maritime accident data written in Japanese, but there's an intent expand its capabilities across multiple industries and languages. 
- The primary hurdle lies in sourcing well-formatted accident and incident reports in different languages. If you're engaged in the Maritime Shipping industry, aviation, or railways, regardless of whether you're in Japan or elsewhere in the world, your contribution to this project would be greatly appreciated. Your collaboration can significantly assist in the development of the system and help extend its linguistic capabilities.
