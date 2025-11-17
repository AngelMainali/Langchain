from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field 


load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation", 
)

model = ChatHuggingFace(llm=llm)

class Person(BaseModel):
    name: str = Field(description = "Name of Person")
    age: int = Field(gt=15, description = "Age of person")
    city: str = Field(description= "Name of city the person belongs to")


parser = PydanticOutputParser(pydantic_object = Person)

template = PromptTemplate(
    template = 'Give the name, age, and city of person from {place} \n {format_instruction}',
    input_variables = ['place'],
    partial_variables = {'format_instruction': parser.get_format_instructions()}
)

# prompt = template.invoke({'place':'Nepal'})

# result = model.invoke(prompt)

# final_result = parser.parse(result.content)

chain = template | model | parser 

final_result = chain.invoke({'place':'China'})

print(final_result)