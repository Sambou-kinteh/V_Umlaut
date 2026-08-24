
from abc import ABC, abstractmethod

class Model(ABC):

    """
    Abstract base class for the different models in this architecture
    """

    # TODO model has class param model
    def __init__(self):...

    @property
    def model(self): ...

    @model.setter
    def model(self, model): ...

    @abstractmethod
    def fit(self): ...

    @abstractmethod
    def project(self): ...

    @abstractmethod
    def non_minimal_refit(self): ...