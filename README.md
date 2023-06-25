# **AutoMarineSI** : A Near-Miss Analyzer using GPT 3 & 4

AutoMarineSI is a near-miss analyzer built upon the capabilities of OpenAI's GPT-3 and GPT-4 models.

The system is designed to intelligently integrate information from a given near-miss report with historical accident data. From this, AutoMarineSI strives to generate scenarios of potential future accidents. Furthermore, it offers suggestions for safety countermeasures based on the analysed near-miss report.

Though initially conceived with maritime navigation at its core, the methodology and adaptability of AutoMarineSI make it a beneficial tool across a range of industries. Beyond maritime, it holds potential for application in domains such as aviation and railways.


## **Pre-Setup**

1. Install the required packages.
2. Create a .env file and specify your own keys and Pinecone environment. The format should be:  
`OPENAI_API_KEY=...`  
`PINECONE_API_KEY=...`  
`PINECONE_ENVIRONMENT=...`  
3. Specify the output language in AutoMarinSI.py. The default is set to "Japanese".
4. Specify the OpenAI model in AutoMarinSI.py. The default models are “gpt3.5-turbo” and “text-embedding-ada-002”.
5. Specify the OpenAI model in AddEmbedding.py. The default model is “text-embedding-ada-002”.
6. Save your accident database (in CSV format) in “data” directory. Refer to the "Format of accident data" section below about the format.


## **Execution**

1. Run "Cal_embedding.py" to add embedding values to your CSV file. (This is required only the first time.)
2. Run "ToPinecone.py" to import the accident database into Pinecone[^1]. (This is also only required the first time.)
3. Execute "AutoMarineSI.py".
4. When you see the prompt "Please input your near-miss report:", proceed to input your report.

[^1]:When attempting to upsert to the Pinecone database, you might occasionally encounter an error message from the Pinecone server, particularly just after creating a new Pinecone index. If this occurs, don't worry. Typically, simply retrying the script resolves the issue. The second or third attempt is usually successful.



## Format and Definition of Accident Data

Cal_embedding.py reads csv file with the following format:  
`“ReportID”,  "Date_accident”,  "Type_accident”,  "Title”,  "Place”,  “URL”,  "Outline”,  "Cause”`  

Please also refer "sample_of_accident-db.csv" in data folder.

These are meaning of each columns:
- ReportID: Unique ID for accident
- Date_accident: Date of the accident
- Type_accident: Type of the accident
- Title: Title of the accident report or Name of the accident
- Place: Place of the accident
- URL: Download URL for the accident report
- Outline: Outline of the accident
- Cause: Cause of the accident



## Sample Data

### How to Use

If you don't have a past accident database, you can use the sample file "JTSB202306.csv" available in this repository. This file was downloaded from the **[Japan Transport Safety Board](https://www.mlit.go.jp/jtsb/index.html)** in June 2023. Please ensure to read the "Copyright of Sample Accident Data" below before using it.

In the file, you'll find the headers are written in Japanese. You'll need to rename these headers and remove some columns from the original file. The script "PickAndRename.py" is designed to perform these necessary actions. Please execute this script after you have saved your OPENAI_API_KEY in your .env file. Once you've run the script, please proceed to step 1 of the Execution section: "Run Cal_embedding.py".


### Copyright

The sample file "JTSB202306.csv" in this repository was obtained from the **[Japan Transport Safety Board](https://www.mlit.go.jp/jtsb/english.html)** in June 2023. You can find the terms of use at **[https://www.mlit.go.jp/jtsb/cyo.html](https://www.mlit.go.jp/jtsb/cyo.html)**. While you are permitted to use this data, you must explicitly state that it was sourced from the **[Japan Transport Safety Board](https://www.mlit.go.jp/jtsb/index.html)** whenever you use it. Additionally, if you modify the data, you must clearly indicate that edits have been made in addition to citing the original source.



## My messages

- While my expertise lies in the Maritime industry, I do not possess the same proficiency in GitHub, Python, or other computer science domains. If you are a software engineer, I would greatly appreciate your assistance.
- If you encounter any bugs or have suggestions to improve the system, I encourage you to make a pull request. All contributions are welcome.
- At present, the system has been tested solely with maritime accident data written in Japanese, but there's an intent expand its capabilities across multiple industries and languages. The primary hurdle lies in sourcing well-formatted accident and incident reports in different languages. If you're engaged in the Maritime Shipping industry, aviation, or railways, regardless of whether you're in Japan or elsewhere in the world, your contribution to this project would be greatly appreciated. Your collaboration can significantly assist in the development of the system and help extend its linguistic capabilities.
- I've written a post that provides an overview of AutoMarineSI. Please take a look if you have any: **[English](https://www.fmcho.com/posts/2023-06-25-2)** | **[日本語](https://www.fmcho.com/posts/2023-06-25-1)**
