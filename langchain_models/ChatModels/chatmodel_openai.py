from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model = 'gpt-4', temperature = 0.3, max_completion_tokens= 100)

result = model.invoke("What is the capital City of Nepal")

print(result) #print all meta data

print(result.content)  # only prints required result 