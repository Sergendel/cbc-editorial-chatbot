from abc import ABC, abstractmethod
from typing import Dict, List


class EmbeddingLoader(ABC):
    @abstractmethod
    def generate_embeddings(
        self,
    ) -> List[Dict]:
        """
        Abstract method defining how embeddings are generated.

        Parameters:
        - data (dict): provided input data (one document/article).

        Returns:
        - List[Dict]: returns list of embeddings with metadata.
        """
        pass
