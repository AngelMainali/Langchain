from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

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

prompt1 = PromptTemplate(
    template = 'Generate Notes from the {text}',
    input_variables = ['text']
)

prompt2 = PromptTemplate(
    template = ' Generate mcq from the following {text}',
    input_variables = ['text']
)

prompt3 = PromptTemplate(
    template = 'Combine the notes and the mcq in a single documents \n notes = {notes}, mcq = {mcq}',
    input_variables = ['notes','mcq']
)

parse = StrOutputParser()

parallel_chain = RunnableParallel({
    'notes': prompt1 | model1 | parse,
    'mcq': prompt2 | model2 | parse
})

merge_chain = prompt3 | model2 | parse 

chain = parallel_chain | merge_chain


text = """
Logistic regression, like linear regression, is a type of linear model that examines the relationship between predictor variables (independent variables) and an output variable (the response, target or dependent variable). The key difference is that linear regression is used when the output is a continuous value—for example, predicting someone's credit score. Logistic regression is used when the outcome is categorical, such as whether a loan is approved or not.

In logistic regression, the model predicts the probability that a specific outcome occurs. For instance, given someone’s financial profile, we might predict the probability that their loan is approved. The output of the model is a value between 0 and 1. Based on a threshold—often 0.5—we classify the outcome as either "approved" or "not approved." Instead of drawing a straight line through the data as we would in linear regression, logistic regression fits an S-shaped curve to map input values to a probability.

Both linear and logistic regression use statistical tests to evaluate which predictor variables meaningfully impact the output. Techniques such as the t-test and analysis of variance (ANOVA) (or likelihood ratio tests for logistic regression) generate p-values for each coefficient, helping us assess whether the relationship is statistically significant. A low p-value (typically below 0.05) suggests that the variable contributes meaningfully to the model. We also evaluate the goodness of fit—how well the model explains the observed outcomes—using different metrics depending on the regression type.  

As we build models, it’s important to guard against overfitting, where the model captures noise in the training data and performs poorly on new data. This risk increases when we have many predictor variables but a small sample size. To address this issue, we can apply regularization, a technique that reduces the influence of less important variables by shrinking their coefficients. Careful attention must also be paid to outliers, as they can distort the model and lead to misleading p-values or coefficients. In practice, we improve models through multiple iterations of feature selection, testing and refinement.

To contrast the two models more concretely, consider a linear regression scenario where we want to predict someone's credit score, based on features like their current savings. We can model this as:

"""

result = chain.invoke({'text':text})

print(result)

chain.get_graph().print_ascii()