from abc import ABC, abstractmethod
from typing import Any, Dict, List


class EmbeddingLoaderBase(ABC):
    @abstractmethod
    def load_embeddings(self, dataholder: Any) -> List[Dict[str, Any]]:
        """
        The interface for embedding loaders.
        """
        pass
