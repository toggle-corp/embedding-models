import json
import os
from enum import Enum
from typing import List, Union

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


MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/gtr-t5-large")
MODEL_TYPE = EmbeddingModelType(int(os.getenv("EMBEDDING_MODEL_TYPE", "1")))
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", None)

embedding_model = None

if MODEL_TYPE == EmbeddingModelType.SENTENCE_TRANSFORMERS:
    embedding_model = SentenceTransformerEmbeddingModel(model=MODEL_NAME)
elif MODEL_TYPE == EmbeddingModelType.OLLAMA:
    embedding_model = OllamaEmbeddingModel(model=MODEL_NAME, base_url=OLLAMA_BASE_URL)
elif MODEL_TYPE == EmbeddingModelType.OPENAI:
    embedding_model = OpenAIEmbeddingModel(model=MODEL_NAME)


class RequestSchemaForEmbeddings(BaseModel):
    """Request Schema"""

    texts: Union[str, List[str]]


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

    if embedding_model:
        if isinstance(item.texts, str):
            return embedding_model.embed_query(text=item.texts)
        elif isinstance(item.texts, list):
            return embedding_model.embed_documents(texts=item.texts)
    return []


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
