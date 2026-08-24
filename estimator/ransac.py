
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

        super().__init__()

    def __next__(self):

        features = next(self.features)      # 5 correspondences, (5, 3, 2)
        self.model.fit(features.reshape(3, 5, 2))            # model will be determined and set to model.model

        # (N, ) vector
        deviations : ndarray = self.model.project(self.features.features)
        deviation_mask : ndarray = deviations[deviations < self.projection_threshold]
        self.features.remove_outliers(deviation_mask)                       # removes outliers

    def __iter__(self): return self
