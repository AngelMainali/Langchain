from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loader = DirectoryLoader(
    path = 'demofolder',
    glob = '*.pdf',
    loader_cls = PyPDFLoader
)

# docs = loader.load()  # load()  --> it loads all docs at once

# print(len(docs))
# print(docs[0].metadata)
# print(docs[0].page_content)

docs = loader.lazy_load()  # lazy_load()  --> fetch one docs at a time 

for document in docs:
    print(document.metadata)

