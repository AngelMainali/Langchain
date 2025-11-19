from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2')

splitter = SemanticChunker(
    embeddings=embeddings,
    breakpoint_threshold_type = 'standard_deviation',
    breakpoint_threshold_amount = 0.2
)

sample = '''
    Artificial Intelligence (AI) is a branch of computer science that focuses on creating machines capable of performing tasks that usually require human intelligence. These tasks include problem-solving, decision-making, and natural language understanding.

Machine Learning (ML) is a subset of AI that allows systems to learn from data and improve their performance over time without explicit programming. It is widely used in applications such as recommendation systems, speech recognition, and autonomous vehicles.

Deep Learning, a further subset of ML, uses neural networks with multiple layers to model complex patterns in large datasets. Techniques like convolutional neural networks (CNNs) and recurrent neural networks (RNNs) have revolutionized fields like computer vision and natural language processing.

AI is being applied across industries including healthcare, finance, education, and transportation. In healthcare, it helps with disease diagnosis and personalized treatment plans. In finance, AI assists in fraud detection and algorithmic trading.

Despite its potential, AI also raises ethical concerns. Issues like data privacy, algorithmic bias, and job displacement need careful consideration to ensure responsible deployment of AI technologies.

'''
docs = splitter.create_documents([sample])

print(len(docs))

print(docs[0].page_content)

print("**************************************************************************")

print(docs[1].page_content)