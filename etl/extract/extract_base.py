from abc import ABC, abstractmethod

class ExtractBase(ABC):
    """
       Abstract base class for data extraction.
       All extractor classes inherit from this class
       and implement the 'extract' method.
       """
    @abstractmethod
    def extract(self):
        pass