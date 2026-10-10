from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

def get_document_loader():
    loader = DirectoryLoader('documents', glob="**/*.pdf", show_progress=True, loader_cls=PyPDFLoader)
    docs = loader.load()
    return docs
