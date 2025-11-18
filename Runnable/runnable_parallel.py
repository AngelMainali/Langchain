from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel

load_dotenv()

llm1 = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation", 
)

llm2 = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation", 
)

model1 = ChatHuggingFace(llm=llm1)

model2 = ChatHuggingFace(llm=llm2)

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template = 'Generate a 2 line facebook post about {topic}',
    input_variables = ['text']
)

prompt2 = PromptTemplate(
    template = 'Generate a 5 line linkedin post about {topic}',
    input_variables = ['text']
)

parallel_chain = RunnableParallel({
    'facebook': RunnableSequence(prompt1, model1, parser),
    'linkedin': RunnableSequence(prompt2, model2, parser),
})

result = parallel_chain.invoke({'topic':'ML'})

print(result)