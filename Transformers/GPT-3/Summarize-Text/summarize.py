import openai
import pdfplumber
import numpy as np


# Converting pdf to text
filepath = ''
paper = pdfplumber.open(filepath).pages

# Function for summarizing the document through openAI API
def summarizeDocument(file):

    openai.api_key = "YOUR API KEY"

    # Looping through document pages to summarize by page
    for page in file:
        text = page.extract_text() + "\n tl;dr:"
        response = openai.Completion.create(
            engine="davinci",
            prompt=text,
            temperature=0.3,
            max_tokens=200,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=["\n"]
        )

        print(response["choices"][0]["text"])

# Summarizing the given paper by calling the function
summarizeDocument(paper)


