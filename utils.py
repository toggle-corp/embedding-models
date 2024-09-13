import json
import logging
import os
from pathlib import Path

from huggingface_hub import snapshot_download

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def download_models(sent_embedding_model: str):
    models_path = Path("/opt/models")
    models_info_path = models_path / "model_info.json"

    if not os.path.exists(models_path):
        os.makedirs(models_path)

    if not any(os.listdir(models_path)):
        logger.info("Downloading the model")
        embedding_model_local_path = snapshot_download(repo_id=sent_embedding_model, cache_dir=models_path)
        models_info = {
            "model": sent_embedding_model,
            "model_path": embedding_model_local_path,
        }

        with open(models_info_path, "w", encoding="utf-8") as m_info_f:
            json.dump(models_info, m_info_f)

    else:
        if os.path.exists(models_info_path):
            logger.info("Models already exists.")
            logger.info(models_info_path)
            with open(models_info_path, "r", encoding="utf-8") as m_info_f:
                models_info = json.load(m_info_f)

    return models_info
