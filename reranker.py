from typing import List

import torch.nn.functional as F
from sentence_transformers import CrossEncoder
from torch import Tensor


def get_scores(query: str, documents: List[str], model_name: str = "cross-encoder/ms-marco-MiniLM-L-2-v2"):
    """Get the scores"""
    model = CrossEncoder(model_name=model_name, max_length=512)
    doc_tuple = [(query, doc) for doc in documents]
    scores = model.predict(doc_tuple)
    return F.softmax(Tensor(scores), dim=0).tolist()
