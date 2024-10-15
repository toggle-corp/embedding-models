import json
import logging
import os
from pathlib import Path

from huggingface_hub import snapshot_download

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def download_model(embedding_model: str, models_path: str):
    """Downloads the model"""
    logger.info("Downloading the model")
    embedding_model_local_path = snapshot_download(repo_id=embedding_model, cache_dir=models_path)
    return embedding_model_local_path


def check_models(sent_embedding_model: str):
    """Check if the model already exists"""
    models_path = Path("/opt/models")
    models_info_path = models_path / "model_info.json"

    if not os.path.exists(models_path):
        os.makedirs(models_path)

    if not any(os.listdir(models_path)):
        embedding_model_local_path = download_model(embedding_model=sent_embedding_model, models_path=models_path)
        models_info = {
            sent_embedding_model: embedding_model_local_path,
        }

        with open(models_info_path, "w", encoding="utf-8") as m_info_f:
            json.dump(models_info, m_info_f)
        return embedding_model_local_path
    if os.path.exists(models_info_path):
        with open(models_info_path, "r", encoding="utf-8") as m_info_f:
            models_info_dict = json.load(m_info_f)
        if sent_embedding_model not in models_info_dict.keys():
            embedding_model_local_path = download_model(embedding_model=sent_embedding_model, models_path=models_path)
            models_info_dict[sent_embedding_model] = embedding_model_local_path
            with open(models_info_path, "w", encoding="utf-8") as m_info_f:
                json.dump(models_info_dict, m_info_f)
            return embedding_model_local_path

        logger.info("Model is already available.")
        return models_info_dict[sent_embedding_model]
