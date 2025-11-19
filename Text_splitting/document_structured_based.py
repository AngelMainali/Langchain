from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader('question_paper.pdf')

docs = loader.load()

splitter = RecursiveCharacterTextSplitter.from_language(
    language = Language.MARKDOWN,  
    chunk_size = 750,
    chunk_overlap = 0
)

chunks = splitter.split_documents(docs)

print(chunks[0].page_content)