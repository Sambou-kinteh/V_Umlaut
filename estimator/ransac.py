
import cv2 as cv
import numpy as np

from numpy import ndarray
from estimator_pipeline import Estimator
from estimator_pipeline import Evaluator
from models.model import Model
from feature_extraction.features import Features


class Ransac(Estimator):

    def __init__(self, N, model: Model, evaluator: Evaluator, features : Features):

        self.__N = N
        self.model = model
        self.evaluator = evaluator
        self.features = features

        super().__init__()

    def __next__(self):

        # todo refit model with rest of inliners
        # todo raise StopIteration

        # TODO vectorize instead of looping

        features = next(self.features)      # 5 correspondences
        self.model.fit(features)            # model will be determined and set to model.model   # TODO ADJUST ARCHITECTURE

        # (N, ) Tensor
        # projection_error < .000001
        # todo later evaluator definied with simple reprojection with varaible threshold

        threshold : float = .000001
        deviations : ndarray = self.model.project()                # TODO IMPLEMENT # model will be projected and will be return for every other correspondence the diviation # todo pass this stage to evaluator
        deviation_mask : ndarray = deviations[deviations < threshold]
        self.features.remove_outliners(deviation_mask)              # removes outliners     # TODO IMPLEMENT

    def __iter__(self): return self
