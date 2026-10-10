from threading import RLock

from services.embeddings import get_embeddings
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from services.chunking import get_parent_child_chunks
from services.document_processor import get_document_loader

documents = get_document_loader()
chunks, parent_documents = get_parent_child_chunks(documents)
model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
_retrieval_lock = RLock()
_shared_retriever = None

def get_advanced_retriever(chunks, vector_db):
    """
    chunks: The list of raw LangChain Document objects you created during chunking.
    vector_db: Your loaded FAISS database.
    """
    # 1. Setup Keyword Search (BM25)
    # We pass the raw chunks so BM25 can build its keyword frequency index
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 5 # Ask BM25 for its top 5 keyword matches

    # 2. Setup Semantic Search (FAISS)
    faiss_retriever = vector_db.as_retriever(search_kwargs={"k": 5}) # Ask FAISS for its top 5 semantic matches

    # 3. Combine them into a Hybrid Search (uses Reciprocal Rank Fusion)
    hybrid_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever], 
        weights=[0.5, 0.5] # Give equal weight to keywords and semantics
    )

    # 4. Setup the Cross-Encoder Re-ranker
    # BAAI/bge-reranker-base is an industry-standard, lightweight reranking model
    compressor = CrossEncoderReranker(model=model, top_n=3) # Only keep the absolute best 3 chunks

    # 5. Wrap the Hybrid Retriever with the Re-ranker
    advanced_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, 
        base_retriever=hybrid_retriever
    )
    
    return advanced_retriever


def initialize_retrieval():
    """Initialize shared models/index before the web server accepts requests."""
    global _shared_retriever
    with _retrieval_lock:
        if _shared_retriever is None:
            _shared_retriever = get_advanced_retriever(chunks, get_embeddings())
        return _shared_retriever


def _retrieve_context(query):
    # Avoid overlapping native model inference from Flask request threads.
    with _retrieval_lock:
        import faiss
        faiss.omp_set_num_threads(1)
        return initialize_retrieval().invoke(query)


def get_retriever():
    db = get_embeddings()
    retriever = db.as_retriever()
    return retriever

def get_claim_approval_context():
    # Run the query through BM25 -> FAISS -> Hybrid Fusion -> Cross-Encoder
    context_docs = _retrieve_context("What are the documents required for claim approval?")
    
    # Extract the text just like before
    claim_approval_context = ""
    for doc in context_docs:
        claim_approval_context += doc.page_content + "\n\n"
        
    return claim_approval_context

def get_general_exclusion_context():
    context = _retrieve_context("Give a list of all general exclusions")

    # This is a complete-list query. Search the child chunks first, then expand
    # the matching General Exclusions child to its non-embedded parent section.
    for document in context:
        if document.metadata.get("section_title", "").upper() == "GENERAL EXCLUSIONS":
            parent = parent_documents.get(document.metadata.get("parent_id"))
            if parent:
                return parent.page_content

    # Safe fallback if reranking misses the section title: return the exact
    # General Exclusions parent rather than an incomplete top-k child list.
    for parent in parent_documents.values():
        if parent.metadata.get("section_title", "").upper() == "GENERAL EXCLUSIONS":
            return parent.page_content

    return ""
