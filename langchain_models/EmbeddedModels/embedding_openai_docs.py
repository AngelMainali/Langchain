from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = OpenAIEmbeddings(model = 'text-embedding-3-large', dimensions = 32)

documents =[
    "Nepal is country",
    "Nepal is democratic country",
    "Nepal is in between China and India"
]

result = embedding.embed_documents(documents)

print(str(result))