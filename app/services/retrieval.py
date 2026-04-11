# app/services/retrieval.py
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from app.settings import CHROMA_DIR

def get_embeddings():
    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001"
    )

def get_vectorstore():
    return Chroma(
        collection_name="books",
        embedding_function=get_embeddings(),
        persist_directory=str(CHROMA_DIR),
    )