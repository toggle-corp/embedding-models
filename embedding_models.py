from dataclasses import dataclass, field
from typing import List

import numpy as np
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from sentence_transformers import SentenceTransformer
from torch import Tensor

from utils import download_models


@dataclass
class SentenceTransformerEmbeddingModel(Embeddings):
    """
    Embedding model using Sentence Transformers
    """

    model: str
    embedding_model: SentenceTransformer = field(init=False)

    def __post_init__(self):
        """
        Post initialization
        """
        models_info = download_models(sent_embedding_model=self.model)
        self.st_embedding_model = SentenceTransformer(model_name_or_path=models_info["model_path"])

    def embed_documents(self, texts: list) -> np.ndarray:
        """
        Generate embeddings for a list of documents
        """
        v_representation = self.st_embedding_model.encode(texts)
        return v_representation.tolist()

    def embed_query(self, text: str) -> np.ndarray:
        """
        Generate embedding for a piece of text
        """
        v_representation = self.st_embedding_model.encode(text)
        return v_representation.tolist()

    def check_similarity(self, embeddings_1: np.ndarray, embeddings_2: np.ndarray) -> Tensor:
        """
        Computes the cosine similarity between two embeddings
        """
        return self.st_embedding_model.similarity(embeddings_1, embeddings_2)

    def get_model(self):
        """Returns the model"""
        return self.st_embedding_model


@dataclass
class OllamaEmbeddingModel(Embeddings):
    """
    Embedding model using Ollama (locally deployed)
    """

    model: str
    base_url: str

    def __post_init__(self):
        """
        Post initialization
        """
        self.ollama_embed_model = OllamaEmbeddings(model=self.model, base_url=self.base_url)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of documents.
        """
        return self.ollama_embed_model.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        """
        Generate embedding for a piece of text
        """
        return self.ollama_embed_model.embed_query(text=text)

    def get_model(self):
        """Returns the model"""
        return self.ollama_embed_model


@dataclass
class OpenAIEmbeddingModel(Embeddings):
    """
    Embedding Model using OpenAI
    """

    model: str

    def __post_init__(self):
        """
        Post initialization
        """
        self.openai_embed_model = OpenAIEmbeddings(model=self.model)

    def embed_documents(self, texts: List[str]):
        """
        Generate embeddings for a list of documents.
        """
        return self.openai_embed_model.embed_documents(texts=texts)

    def embed_query(self, text: str):
        """
        Generate embedding for a piece of text
        """
        return self.openai_embed_model.embed_query(text=text)

    def get_model(self):
        """Returns the model"""
        return self.openai_embed_model
