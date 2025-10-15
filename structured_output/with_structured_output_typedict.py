from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from typing import _TypedDict

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation", 
)

model = ChatHuggingFace(llm=llm)

class Review(_TypedDict):
    summary : str
    sentiment : str


structured_model = model.with_structured_output(Review)    


result =structured_model.invoke("""I recently bought the Apple AirPods Pro 2, and I’m really impressed. The noise cancellation is excellent, the sound quality is crisp, and they fit comfortably even after hours of use. The only downside is the price — it’s quite expensive. But overall, they feel premium and deliver great performance.""")

print(result)
print(result['summary'])
print(result['sentiment'])