from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

chat_template = ChatPromptTemplate([
     ('system','You are a helpful {domain} expert' ),
    ('human', 'Explain in simple terms, what is {topic}')

    # Doesnt work in this way dynamic template in multi-turn message

    # SystemMessage(content = 'You are a helpful {domain} expert'),
    # HumanMessage(content = 'Explain in simple terms, what is {topic}')
])

prompt = chat_template.invoke({'domain': 'cricket', 'topic': 'spin'})

print(prompt)