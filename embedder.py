# embedder.py

import torch
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

def create_vectorstore_from_text(text):
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = [Document(page_content=chunk) for chunk in text_splitter.split_text(text)]

    # Try GPU, fall back to CPU
    #device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu" 
    
    try:
        # Try to pre-allocate a small tensor to test GPU memory availability
        torch.zeros((1,), device=device)
        print("Loading embeddings on GPU...")
    except RuntimeError:
        device = "cpu"
        print("Insufficient GPU memory, switching to CPU for embeddings.")

    # Load embeddings with the selected device
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": device}
    )

    # Create FAISS vectorstore
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore
