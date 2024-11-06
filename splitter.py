from typing import List

from langchain.schema import Document
from langchain.text_splitter import SentenceTransformersTokenTextSplitter


def langchain_document_to_dict(doc: Document):
    """
    Converts langchain Document to dictionary
    """
    return {"page_content": doc.page_content, "metadata": doc.metadata}


def dict_to_langchain_document(doc: dict):
    """
    Converts dictionary to Langchain docuemnt
    """
    return Document(page_content=doc["page_content"], metadata=doc["metadata"])


def get_split_documents_using_token_based(model_name: str, documents: List[dict], chunk_size: int, chunk_overlap: int):
    """
    Splits documents into multiple chunks using Sentence Transformer
    token based.
    """
    splitter = SentenceTransformersTokenTextSplitter(
        chunk_overlap=chunk_overlap, model_name=model_name, tokens_per_chunk=chunk_size
    )
    langchain_docs = [dict_to_langchain_document(d) for d in documents]
    splitted_docs = splitter.split_documents(documents=langchain_docs)
    return [langchain_document_to_dict(d) for d in splitted_docs]
