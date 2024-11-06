import json
from enum import Enum
from typing import List, Optional, Union

from dotenv import load_dotenv
from fastapi import FastAPI, Response, status
from pydantic import BaseModel

from embedding_models import (
    OllamaEmbeddingModel,
    OpenAIEmbeddingModel,
    SentenceTransformerEmbeddingModel,
)
from reranker import get_scores

# from langchain.schema import Document
from splitter import get_split_documents_using_token_based

load_dotenv()

app = FastAPI()


class EmbeddingModelType(Enum):
    """
    Embedding model types
    """

    SENTENCE_TRANSFORMERS = 1
    OLLAMA = 2
    OPENAI = 3


class RequestSchemaForEmbeddings(BaseModel):
    """Request Schema"""

    type_model: EmbeddingModelType
    name_model: str
    texts: Union[str, List[str]]
    base_url: Optional[str] = None


class RequestSchemaForTextSplitter(BaseModel):
    """Request Schema"""

    model: str
    documents: str
    chunk_size: int
    chunk_overlap: int


class RequestSchemaForReRankers(BaseModel):
    """Request Schema"""

    query: str
    documents: List[str]


@app.get("/")
async def home():
    """Returns a message"""
    return Response(content="Embedding handler using models for texts", status_code=status.HTTP_200_OK)


@app.post("/get_embeddings")
async def generate_embeddings(item: RequestSchemaForEmbeddings):
    """
    Generates the embedding vectors for the text/documents
    based on different models
    """
    type_model = item.type_model
    name_model = item.name_model
    base_url = item.base_url
    texts = item.texts

    def generate(em_model, texts):
        if isinstance(texts, str):
            return em_model.embed_query(text=texts)
        elif isinstance(texts, list):
            return em_model.embed_documents(texts=texts)
        return None

    if type_model == EmbeddingModelType.SENTENCE_TRANSFORMERS:
        embedding_model = SentenceTransformerEmbeddingModel(model=name_model)
        return generate(em_model=embedding_model, texts=texts)

    elif type_model == EmbeddingModelType.OLLAMA:
        embedding_model = OllamaEmbeddingModel(model=name_model, base_url=base_url)
        return generate(em_model=embedding_model, texts=texts)

    elif type_model == EmbeddingModelType.OPENAI:
        embedding_model = OpenAIEmbeddingModel(model=name_model)
        return generate(em_model=embedding_model, texts=texts)


@app.post("/split_docs_based_on_tokens")
async def get_split_docs(item: RequestSchemaForTextSplitter):
    """Splits the documents using the model tokenization method"""
    docs = json.loads(item.documents)
    return get_split_documents_using_token_based(
        model_name=item.model, documents=docs, chunk_size=item.chunk_size, chunk_overlap=item.chunk_overlap
    )


@app.post("/docs_reranking_scores")
async def get_reranked_docs(item: RequestSchemaForReRankers):
    """Get reranked documents"""
    return get_scores(item.query, item.documents)
