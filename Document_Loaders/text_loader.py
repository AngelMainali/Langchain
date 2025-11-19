from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation", 
)

model = ChatHuggingFace(llm=llm)

prompt = PromptTemplate(
    template = ' write a 2 line  summary for give {text}',
    input_variables = ['text']
)

parser = StrOutputParser()

loader = TextLoader('ml_langchain.txt', encoding='utf-8')

docs = loader.load()


print(docs[0])

print("***************************************************************************************")

print(docs[0].page_content)

print("***************************************************************************************")

print(docs[0].metadata)

print("***************************************************************************************")

print(type(docs))

print("***************************************************************************************")

chain = prompt | model | parser

result = chain.invoke({'text':docs[0].page_content})

print(result)

print("***************************************************************************************")