import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document  # ✅ Required for FAISS indexing
 
def load_vectorstore(persist_path="vectorstore"):

    """Loads a FAISS vector store or creates one if missing."""

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # ✅ Check if vector store exists
    if not os.path.exists(f"{persist_path}.faiss") or not os.path.exists(f"{persist_path}.pkl"):
        print("❗ Vector store not found, creating a new one...")
        # ✅ Initialize with a test document to prevent errors

        empty_documents = [Document(page_content="Placeholder document for FAISS.")]
        vectorstore = FAISS.from_documents(empty_documents, embeddings)
        vectorstore.save_local(persist_path)
 
    try:

        # ✅ Load existing FAISS index
        vectorstore = FAISS.load_local(
            persist_path,
            embeddings,
            allow_dangerous_deserialization=True
        )
    except Exception as e:
        print(f"❗ Failed to load FAISS: {e}")

        return None
 
    return vectorstore
 
def retrieve_relevant_documents(query, vectorstore, k=4):
    """Retrieves relevant documents using FAISS similarity search."""

    if vectorstore is None:
        return "❗ Vector store not loaded correctly."
 
    docs = vectorstore.similarity_search(query, k=k)
    return "\n\n".join([doc.page_content for doc in docs]) if docs else "❗ No relevant documents found."
 
