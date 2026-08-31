
import numpy as np

from numpy import ndarray
from estimator.estimator_pipeline import Estimator
from estimator.estimator_pipeline import Evaluator
from models.model import Model
from feature_extraction.features import Features


class Ransac(Estimator):

    def __init__(self, model: Model, features : Features,
                 projection_threshold : float = 1e-4,
                 loss_metric : str = Evaluator.LOSS_METRIX_SAMPSON, useLO = True):

        self.model = model
        self.loss_metric = loss_metric
        self.features = features
        self.projection_threshold = projection_threshold
        self.useLO = useLO

        self.iters_after_best : int = 0
        self.best_inlier_count : int = 0
        self.best_inlier_mask : ndarray
        self.best_point_confi : ndarray

        super().__init__()

    def __next__(self):

        self.iters_after_best += 1
        features, features_index = next(self.features)      # 5 correspondences, (5, 3, 2)
        current_model = self.model.model
        self.model.fit(features.transpose(1, 0, 2))            # model will be determined and set to model.model

        if current_model is not None:       # captures the case that the first ever point config invalid
            # (N, ) vector
            deviations : ndarray = Evaluator.reprojection(self.model.model, self.features.features[..., 0], self.features.features[..., 1]) \
                if self.loss_metric != Evaluator.LOSS_METRIX_SAMPSON \
                else Evaluator.sampson_distance(self.model.model, self.features.features)

            inlier_mask : ndarray = np.abs(deviations) < self.projection_threshold       # Inlier - True, Outlier - False
            outlier_mask : ndarray = np.abs(deviations) >= self.projection_threshold * 10000       # Inlier - False, Outlier - True
            self.features.outlier_mask = outlier_mask & self.features.outlier_mask if self.features.outlier_mask is not None else outlier_mask

            inlier_count : int = np.count_nonzero(inlier_mask)
            if inlier_count > self.best_inlier_count:

                self.iters_after_best = 0
                self.best_point_confi = features
                self.best_inlier_mask = inlier_mask
                self.best_inlier_count = inlier_count

                if self.useLO:

                    rng = np.random.default_rng()
                    rand_sij = rng.uniform(.4, .6, size=10)
                    rand_sij = np.round(rand_sij, 3)
                    # rand_sij = rng.uniform(np.nextafter(0,1), 1, size=5)

                    # rand_idx = np.random.choice(len(self.model.good_values), 10)
                    # rand_sij = np.array(self.model.good_values)[rand_idx]

                    for each in rand_sij:

                        self.model.v_umlaut_sij = each
                        self.__local_optimization(each)

            else: self.model.model = current_model

    def __local_optimization(self, new_sij):

        inlier_points = self.features.features[self.best_inlier_mask]  # (inliers, 3, 2)

        model_before = self.model.model.copy()
        previous_sij = self.model.v_umlaut_sij

        self.model.v_umlaut_sij = new_sij
        self.model.non_minimal_refit(inlier_points.transpose(1, 0, 2))

        deviations = Evaluator.sampson_distance(self.model.model, self.features.features)
        refined_mask = np.abs(deviations) < self.projection_threshold
        refined_count = np.count_nonzero(refined_mask)

        if refined_count > self.best_inlier_count:
            self.best_inlier_mask = refined_mask
            self.best_inlier_count = refined_count
        else:
            self.model.v_umlaut_sij = previous_sij
            self.model.model = model_before

    def __iter__(self): return self
