

from __future__ import annotations

import cv2 as cv

from numpy import ndarray
from typing import Final
from MyHelpers.Frame import Frame
from data.inn.synthetic.synthetic_data import *


class Features:

    EXTRACTOR_SIFT : Final = 0
    EXTRACTOR_ORB : Final = 1
    EXTRACTOR_SURF : Final = 2
    EXTRACTOR_SYNTHETIC : Final = 3

    def __init__(self, n : int, N : int, frame1 : Frame, frame2 : Frame, extractor : int, **kwargs):

        self.__features : ndarray = None
        self.__ground_truth : ndarray = None
        self.__n : int = n
        self.__frame1 = frame1
        self.__frame2 = frame2
        self.outlier_mask = None

        if extractor == self.EXTRACTOR_SIFT: self.features = self.__sift(N)
        elif extractor == self.EXTRACTOR_ORB: self.features = self.__orb(N)
        elif extractor == self.EXTRACTOR_SURF: self.features = self.__surf(N)
        elif extractor == self.EXTRACTOR_SYNTHETIC:

            seed = kwargs.get("seed", None)
            sigma = kwargs.get("sigma", 0.0)
            outliers = kwargs.get("outliers", 0)
            self.features = self.__synthetic(seed, N, sigma, outliers)

    @property
    def ground_truth(self) -> ndarray: return self.__ground_truth

    @ground_truth.setter
    def ground_truth(self, ground_truth : ndarray): self.__ground_truth = ground_truth

    @property
    def features(self):
        return self.__features      # (<=N, 3, 2)

    @features.setter
    def features(self, features):

        assert isinstance(features, ndarray), "Invalid feature type"
        self.__features = features

    def remove_outliers(self, inlier_mask : ndarray|None):

        if inlier_mask is not None: self.features = self.features[inlier_mask, ...]

    def __next__(self) -> tuple:

        if self.__features.shape[0] < self.__n: raise StopIteration("Not enough points to continue")

        new_features = self.features.copy()[~self.outlier_mask if self.outlier_mask is not None else ...]
        sample_row_indices = np.random.choice(new_features.shape[0], size=self.__n, replace=True)   # replace = True for repeating points
        return new_features[sample_row_indices, ...], sample_row_indices     # (n, 3, 2)

    def __iter__(self): return self


    def __sift(self, N : int, threshold : float = .75) -> ndarray:

        # TODO add min distance filtering

        #---------- definition and invoking
        sift = cv.SIFT.create(
            nfeatures=N,
        )

        kp1, descr1 = sift.detectAndCompute(self.__frame1, None)
        kp2, descr2 = sift.detectAndCompute(self.__frame2, None)

        index_params = {
            "algorithm": 1,
            "trees": 5
        }
        search_params = {
            "checks": 50
        }
        flann = cv.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(descr1, descr2, k=2)

        #---------- extraction and processing
        points1 = []
        points2 = []

        for i, j in matches:
            if i.distance < threshold * j.distance:
                points1.append(kp1[i.queryIdx].pt)
                points2.append(kp2[i.trainIdx].pt)

        points1 = np.float32(points1)   # (N, 2)
        points2 = np.float32(points2)   # (N, 2)

        points1_homogenous = np.column_stack([points1, np.ones(len(points1), dtype=np.float32)])  # (N, 3)
        points2_homogenous = np.column_stack([points2, np.ones(len(points2), dtype=np.float32)])  # (N, 3)

        return np.dstack([points1_homogenous, points2_homogenous])        # (N, 3, 2)


    def __orb(self, amount : int) -> ndarray: ...
    def __surf(self, amount : int) -> ndarray: ...

    def __synthetic(self, seed : int = None, N : int = 100, sigma : float =  0.0, outliers : int = 0) -> ndarray:

        rng = np.random.default_rng(seed)
        scene = generate(rng, mode=Mode.GENERAL, n_points=N, independent_noise_sigma=sigma)
        scene = inject_outliers(scene, n_outliers=outliers, rng=rng)
        self.ground_truth = scene.fundamental_matrix_ground_truth()

        return scene.to_features()
