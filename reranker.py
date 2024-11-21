from typing import List

import torch.nn.functional as F
from sentence_transformers import CrossEncoder
from torch import Tensor

cross_encoder_model = CrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-2-v2", max_length=512)


def get_scores(query: str, documents: List[str]):
    """Get the scores"""
    doc_tuple = [(query, doc) for doc in documents]
    scores = cross_encoder_model.predict(doc_tuple)
    return F.softmax(Tensor(scores), dim=0).tolist()
