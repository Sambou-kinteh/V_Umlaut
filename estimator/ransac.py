
import numpy as np

from numpy import ndarray
from estimator.estimator_pipeline import Estimator
from estimator.estimator_pipeline import Evaluator
from models.model import Model
from feature_extraction.features import Features


class Ransac(Estimator):

    def __init__(self, model: Model, evaluator: Evaluator|None, features : Features, projection_threshold : float = .000001):

        self.model = model
        self.evaluator = evaluator
        self.features = features
        self.projection_threshold = projection_threshold

        self.best_inlier_count : int = 0
        self.best_inlier_mask : ndarray
        self.best_point_configuration : ndarray

        super().__init__()

    def __next__(self):

        features = next(self.features)      # 5 correspondences, (5, 3, 2)
        current_model = self.model.model
        self.model.fit(features.transpose(1, 0, 2))            # model will be determined and set to model.model

        if current_model is not None:       # captures the case that the first ever point config invalid
            # (N, ) vector
            deviations : ndarray = self.model.project(self.features.features)
            inlier_mask : ndarray = np.abs(deviations) < self.projection_threshold       # Inlier - True, Outlier - False

            inlier_count : int = np.count_nonzero(inlier_mask)
            if inlier_count > self.best_inlier_count:

                print("inlier count: ", inlier_count)
                self.best_point_configuration = features
                self.best_inlier_mask = inlier_mask
                self.best_inlier_count = inlier_count

            else: self.model.model = current_model

    def __iter__(self): return self
