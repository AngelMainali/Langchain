from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import PromptTemplate, load_prompt

load_dotenv()
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation", 
)

model = ChatHuggingFace(llm=llm)

st.header("Research Assistant")


papers = [
    "Attention Is All You Need - Vaswani et al., 2017",
    "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding - Devlin et al., 2019",
    "GPT-3: Language Models are Few-Shot Learners - Brown et al., 2020",
    "ImageNet Classification with Deep Convolutional Neural Networks - Krizhevsky et al., 2012",
    "Deep Residual Learning for Image Recognition (ResNet) - He et al., 2015",
    "YOLOv3: An Incremental Improvement - Redmon & Farhadi, 2018",
    "Graph Neural Networks: A Review of Methods and Applications - Zhou et al., 2018",
    "Reinforcement Learning: An Introduction - Sutton & Barto, 2018",
    "AlphaGo: Mastering the Game of Go with Deep Neural Networks and Tree Search - Silver et al., 2016",
    "VAE: Auto-Encoding Variational Bayes - Kingma & Welling, 2013",
    "GANs: Generative Adversarial Nets - Goodfellow et al., 2014",
    "Transformers for Time Series Forecasting - Lim et al., 2021",
    "Neural Machine Translation by Jointly Learning to Align and Translate - Bahdanau et al., 2015",
    "Speech Recognition with Deep Recurrent Neural Networks - Graves et al., 2013",
    "StyleGAN: A Style-Based Generator Architecture for Generative Adversarial Networks - Karras et al., 2019"
]

explanation_styles = [
    "Beginner-Friendly",
    "Technical",
    "Code-Oriented",
    "Mathematical"
]

explanation_lengths = [
    "Short",
    "Medium",
    "Long"
]


selected_paper = st.selectbox("Select a research paper:", papers)
selected_style = st.selectbox("Select explanation style:", explanation_styles)
selected_length = st.selectbox("Select explanation length:", explanation_lengths)

  
template = load_prompt('template.json')





if st.button('Summarize'):
   chain = template | model 
   result = chain.invoke({
       'selected_paper' : selected_paper,
       'selected_style' : selected_style,
       'selected_length' : selected_length
   })
   st.write(result.content)
