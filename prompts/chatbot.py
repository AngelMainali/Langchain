from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation", 
)

model = ChatHuggingFace(llm=llm)

# This chatbot doesnt capture context (no history is saved)

""""
while True:
    user_input = input('you: ')

    if user_input == 'exit':
        break
    result = model.invoke(user_input)
    print("AI: ", result.content)

""" 


# This chatbot saves history but doesnt identify which message is from user and AI

"""
chat_history = []

while True:
    user_input = input('you: ')
    chat_history.append(user_input)
    
    if user_input == 'exit':
        break
    result = model.invoke(chat_history)
    chat_history.append(result.content)
    print("AI: ", result.content)

print(chat_history) 

"""

# This chatbot overcomes above two issues.

chat_history = [
    SystemMessage(content = "you are helpful assistant ")
]

while True:
    user_input = input('you: ')
    chat_history.append(HumanMessage(user_input))
    
    if user_input == 'exit':
        break
    result = model.invoke(chat_history)
    chat_history.append(AIMessage(result.content))
    print("AI: ", result.content)

print(chat_history) 