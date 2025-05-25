from abc import ABC, abstractmethod


class TransformBase(ABC):
    @abstractmethod
    def transform(self):
        pass
