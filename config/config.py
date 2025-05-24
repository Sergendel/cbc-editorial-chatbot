# config.py
import yaml
from pathlib import Path

class Config:
    def __init__(self, config_file):
        self.project_root = Path(__file__).parent.parent.resolve()  # CBC/ as explicit project root
        with open(config_file, 'r', encoding='utf-8') as f:
            self.cfg = yaml.safe_load(f)

    @property
    def raw_news_path(self):
        return self.project_root / Path(self.cfg['data']['raw_news_path'])

    @property
    def processed_news_path(self):
        return self.project_root / Path(self.cfg['data']['processed_news_path'])

    @property
    def guidelines_url(self):
        return self.cfg['data']['guidelines_url']

    @property
    def processed_guidelines_path(self):
        return self.project_root / Path(self.cfg['data']['processed_guidelines_path'])

    @property
    def embeddings_model(self):
        return self.cfg['embeddings']['model_name']

    @property
    def faiss_index_path(self):
        return self.project_root / Path(self.cfg['embeddings']['faiss_index_path'])

    @property
    def hf_model_name(self):
        return self.cfg['model']['hf_model_name']
