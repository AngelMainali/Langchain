from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field 
from typing import Literal
from langchain_core.runnables import RunnableParallel, RunnableBranch, RunnableLambda

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation", 
    max_new_tokens=50
)

model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

class Feedback(BaseModel):
    sentiment: Literal['positive', 'negative'] = Field(description = "give the sentiment of feedback")

parser2 = PydanticOutputParser(pydantic_object= Feedback)    

prompt1 = PromptTemplate(
    template = 'Distinguish the feedback whether it is positive or negative \n {feedback} \n {format_instruction}',
    input_variables = ['feedback'],
    partial_variables = {'format_instruction':parser2.get_format_instructions()}
)



classifier_chain = prompt1 | model | parser2



prompt2 = PromptTemplate(
    template = 'write proper response to this positive feedback \n {feedback}',
    input_variables = ['feedback'],
)

prompt3 = PromptTemplate(
    template = 'write proper response to this negative feedback \n {feedback}',
    input_variables = ['feedback'],
)


branch_chain = RunnableBranch(
    (lambda x:x['sentiment'] == 'positive', prompt2 | model | parser),
    (lambda x:x['sentiment'] == 'negative', prompt3 | model | parser),
    RunnableLambda(lambda x:"couldn't find sentiment")
)

chain = classifier_chain | branch_chain
 
# feedback = ''' the cost of iphone is too high but the performance is better '''

result = chain.invoke({'feedback':'the cost of iphone is too high'})

print(result)
