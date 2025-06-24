from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings

# Load docs
loader = TextLoader("data.txt")
docs = loader.load()

# Split
splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=50)
splits = splitter.split_documents(docs)

# Embedding & Vector Store
embedding = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(splits, embedding)

# Retrieval Chain
from langchain.chains import RetrievalQA
retriever = vectorstore.as_retriever()

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

response = qa_chain.run("What is LangChain?")
print(response)
