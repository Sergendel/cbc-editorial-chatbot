from pathlib import Path

import yaml


class Config:
    def __init__(self, config_file):
        with open(config_file, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

    @property
    def raw_news_path(self):
        return Path(self.cfg["data"]["raw_news_path"])

    @property
    def processed_news_path(self):
        return Path(self.cfg["data"]["processed_news_path"])

    @property
    def guidelines_url(self):
        return self.cfg["data"]["guidelines_url"]

    @property
    def processed_guidelines_path(self):
        return Path(self.cfg["data"]["processed_guidelines_path"])

    @property
    def embeddings_model(self):
        return self.cfg["embeddings"]["model_name"]

    @property
    def faiss_index_path(self):
        return Path(self.cfg["embeddings"]["faiss_index_path"])

    @property
    def hf_model_name(self):
        return self.cfg["model"]["hf_model_name"]
