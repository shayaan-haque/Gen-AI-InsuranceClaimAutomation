from services.document_processor import get_document_loader
from langchain_community.embeddings import HuggingFaceEmbeddings
from services.chunking import get_text_chunks
from langchain_community.vectorstores import FAISS

def get_embeddings():
    documents = get_document_loader()
    chunks = get_text_chunks(documents)
    db = FAISS.from_documents(
        chunks, HuggingFaceEmbeddings()
    )
    return db
