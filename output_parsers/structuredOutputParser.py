# StructuredOutputParser is deprecated
# ResponseSchema is removed

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation", 
)

model = ChatHuggingFace(llm=llm)

schema =[
    ResponseSchema(name = 'fact_1', description = 'fact 1 about topic'),
    ResponseSchema(name = 'fact_2', description = 'fact 2 about topic'),
    ResponseSchema(name = 'fact_3', description = 'fact 3 about topic')
]

parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template = 'Give three facts about {topic} \n {format_instruction}',
    input_varibales = ['topic'],
    partial_varibales = {'format_instructions': parser.get_format_instructions()}
)

prompt = template.invoke({'topic': 'Nepal'})

result = model.invoke(prompt)

final_result = parser.parse(result.content)

print(final_result)

