import os
from threading import Lock

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from services.document_processor import get_document_loader
from langchain_community.embeddings import HuggingFaceEmbeddings
from services.chunking import get_text_chunks
from langchain_community.vectorstores import FAISS

_embedding_lock = Lock()
_vector_db = None


def get_embeddings():
    """Build once per process, rather than reload models for each claim."""
    global _vector_db
    with _embedding_lock:
        if _vector_db is None:
            import faiss
            import torch

            # Keep native CPU inference conservative in the local Flask app.
            faiss.omp_set_num_threads(1)
            torch.set_num_threads(1)
            documents = get_document_loader()
            chunks = get_text_chunks(documents)
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-mpnet-base-v2",
                model_kwargs={"device": "cpu"},
                encode_kwargs={"batch_size": 16},
            )
            _vector_db = FAISS.from_documents(chunks, embeddings)
        return _vector_db
