from enum import Enum
from typing import List, Optional, Union

from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

from embedding_models import (
    OllamaEmbeddingModel,
    OpenAIEmbeddingModel,
    SentenceTransformerEmbeddingModel,
)

load_dotenv()

app = FastAPI()


class EmbeddingModelType(Enum):
    SENTENCE_TRANSFORMERS = 1
    OLLAMA = 2
    OPENAI = 3


class RequestSchemaForEmbeddings(BaseModel):
    """Request Schema"""

    type_model: EmbeddingModelType
    name_model: str
    texts: Union[str, List[str]]
    base_url: Optional[str] = None


class RequestSchemaForModel(BaseModel):
    """Request Schema"""

    type_model: EmbeddingModelType
    name_model: str
    base_url: Optional[str]


@app.get("/")
async def home():
    return "Embedding handler using models for texts"


@app.post("/get_embeddings")
async def generate_embeddings(item: RequestSchemaForEmbeddings):
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
        embedding_model = OllamaEmbeddingModel(
            model=name_model, base_url=base_url
        )
        return generate(em_model=embedding_model, texts=texts)

    elif type_model == EmbeddingModelType.OPENAI:
        embedding_model = OpenAIEmbeddingModel(model=name_model)
        return generate(em_model=embedding_model, texts=texts)


@app.get("/get_model")
async def get_model(item: RequestSchemaForModel):
    type_model = item.type_model
    name_model = item.name_model
    base_url = item.base_url

    if type_model == EmbeddingModelType.SENTENCE_TRANSFORMERS:
        embedding_model = SentenceTransformerEmbeddingModel(model=name_model)
        return embedding_model.get_model()
    elif type_model == EmbeddingModelType.OLLAMA:
        embedding_model = OllamaEmbeddingModel(
            model=name_model, base_url=base_url
        )
        return embedding_model.get_model()
    elif type_model == EmbeddingModelType.OPENAI:
        embedding_model = OpenAIEmbeddingModel(model=name_model)
        return embedding_model.get_model()
