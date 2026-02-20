import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
import json

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

prompt = PromptTemplate(
    input_variables=["text"],
    template="""
You are a sentiment analysis assistant.

Analyze the sentiment of the following text.

Follow these steps:
1. Identify emotional or opinion words.
2. Decide if the sentiment is positive, negative, or neutral.
3. Estimate a confidence score between 0 and 1.
4. Explain briefly why this sentiment was chosen.

Return the result ONLY as raw JSON (no markdown, no backticks) in this format:
{{
  "text": "<original text>",
  "sentiment": "<positive|negative|neutral>",
  "confidence": <number>,
  "reasoning": "<short explanation>"
}}

Text: {text}
"""
)

user_text = "This course exceeded my expectations. I learned a lot."

final_prompt = prompt.format(text=user_text)

try:
    response = llm.invoke(final_prompt)
    print(response.content)

    cleaned = response.content.strip()

    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1]
        cleaned = cleaned.replace("json", "").strip()

    result = json.loads(cleaned)

    with open("sentiment_result.json", "w") as f:
        json.dump(result, f, indent=4)

    print("Saved to sentiment_result.json")

except Exception as e:
    print("An error occurred:", e)
