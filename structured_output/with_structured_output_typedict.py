from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv
from typing import TypedDict
import json

load_dotenv()

# Configure the endpoint with proper parameters
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    max_new_tokens=256,
    temperature=0.1,
    top_p=0.9,
)

model = ChatHuggingFace(llm=llm)

class Review(TypedDict):
    summary: str
    sentiment: str

# Create a prompt that guides the model to output structured JSON
prompt = ChatPromptTemplate.from_template("""
Extract the following information from the product review as a JSON object with exactly these two keys: "summary" and "sentiment".

Review: {input}

Respond with ONLY valid JSON, no other text:

{{
  "summary": "brief summary here",
  "sentiment": "positive/negative/neutral"
}}
""")

# Create the chain with JSON parser
chain = prompt | model | JsonOutputParser()

result = chain.invoke({
    "input": """I recently bought the Apple AirPods Pro 2, and I'm really impressed. The noise cancellation is excellent, the sound quality is crisp, and they fit comfortably even after hours of use. The only downside is the price — it's quite expensive. But overall, they feel premium and deliver great performance."""
})

print("Full result:", result)
print("Summary:", result['summary'])
print("Sentiment:", result['sentiment'])