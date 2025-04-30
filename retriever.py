# retriever.py
import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

def load_vectorstore(persist_path="vectorstore"):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Allow dangerous deserialization because we trust our own saved FAISS file
    vectorstore = FAISS.load_local(
        persist_path,
        embeddings,
        allow_dangerous_deserialization=True
    )
    return vectorstore

def retrieve_relevant_documents(query, vectorstore, k=4):
    docs = vectorstore.similarity_search(query, k=k)
    return "\n\n".join([doc.page_content for doc in docs])
