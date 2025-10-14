from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2')

# text = 'Kathmandu is the Capital of Nepal'

# vector = embedding.embed_query(text)

documents =[
    "Nepal is country",
    "Nepal is democratic country",
    "Nepal is in between China and India"
]

vector = embedding.embed_documents(documents)

print(str(vector))