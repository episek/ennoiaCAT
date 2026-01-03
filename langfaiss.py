from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

# 1. Some example texts
texts = [
    "OpenAI develops artificial intelligence.",
    "FAISS is a library for efficient similarity search.",
    "Streamlit is used for data apps.",
]

# 2. Create an embeddings model
embedding_model = OpenAIEmbeddings()

# 3. Build a FAISS vector store
db = FAISS.from_texts(texts, embedding_model)

# 4. Save the vector store locally
db.save_local("vectorstore")

print("✅ Vectorstore saved to ./vectorstore/")
