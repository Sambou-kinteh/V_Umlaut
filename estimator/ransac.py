
import cv2 as cv
import numpy as np

from estimator_pipeline import Estimator
from estimator_pipeline import Evaluator
from ..models.model import Model


class Ransac(Estimator):

    def __init__(self, model: Model, evaluator: Evaluator):

        self.model = model
        self.evaluator = evaluator

        super().__init__()

    def model(self): ...
    def fit(self): ...
    def refit(self): ...
    def early_non_minimal_fit(self): ...
    def evaluator(self): ...

    def __next__(self): ...
    def __iter__(self): return self
