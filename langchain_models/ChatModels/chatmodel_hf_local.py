
import os

os.environ['HF_HOME'] = 'D:/huggingface_cache'

from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from transformers import pipeline

""""
gen_pipeline = pipeline(  
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    )


llm = HuggingFacePipeline(pipeline = gen_pipeline,  pipeline_kwargs={"temperature": 0.6, "max_new_tokens": 100})

model = ChatHuggingFace(llm=llm)

result = model.invoke("What is capital of Nepal and India")

print(result.content)

"""

llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task =  "text-generation",
    pipeline_kwargs=dict(
        temperature = 0.6,
        max_new_tokens = 100
    )
)

model = ChatHuggingFace(llm=llm)

result = model.invoke("What is capital of Nepal and India")

print(result.content)