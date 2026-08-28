
import numpy as np
import cv2 as cv

from numpy import ndarray
from typing import Final
from MyHelpers.Frame import Frame


class Features:

    EXTRACTOR_SIFT : Final = 0
    EXTRACTOR_ORB : Final = 1
    EXTRACTOR_SURF : Final = 2

    def __init__(self, n : int, N : int, frame1 : Frame, frame2 : Frame, minPxDistance : int, extractor : int):

        self.__features : ndarray = None
        self.__n : int = n
        self.__frame1 = frame1
        self.__frame2 = frame2

        if extractor == self.EXTRACTOR_SIFT: self.features = self.__sift(N, minPxDistance)
        elif extractor == self.EXTRACTOR_ORB: self.features = self.__orb(N, minPxDistance)
        elif extractor == self.EXTRACTOR_SURF: self.features = self.__surf(N, minPxDistance)


    @property
    def features(self):
        return self.__features      # (<=N, 3, 2)

    @features.setter
    def features(self, features):

        assert isinstance(features, ndarray), "Invalid feature type"
        self.__features = features

    def remove_outliers(self, inlier_mask : ndarray|None):

        if inlier_mask is not None: self.features = self.__features[inlier_mask, ...]

    def __next__(self) -> ndarray:

        if self.__features.shape[0] < self.__n: raise StopIteration("Not enough points to continue")

        sample_row_indices = np.random.choice(self.__features.shape[0], size=self.__n, replace=False)   # replace = True for repeating points
        return self.__features[sample_row_indices, ...]     # (n, 3, 2)

        # for example in the paper
        # X = np.zeros((4, 7))
        # X[0, :] = (1, 0, 0, 1, 2, 1/4, 1/2)
        # X[1, :] = (0, 1, 0, 1, 3, 3/4, 0)
        # X[2, :] = (0, 0, 1, 1, 1, 0, 1/2)
        # X[3, :] = 1
        #
        # P = np.column_stack((np.identity(3), (0, 0, 0)))
        #
        # Q = np.zeros((3, 4))
        # Q[0, :] = (-18/299, 5/299, 5/299, -5/299)
        # Q[1, :] = (1/483, -22/483, 1/483, -1/483)
        # Q[2, :] = (6/253, 6/253, -17/253, -6/253)

        # return np.dstack([P @ X, Q @ X])        # (3, 7, 2)

    def __iter__(self): return self


    def __sift(self, N : int, minDistance : int, threshold : float = .75) -> ndarray:

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


    def __orb(self, amount : int, minDistance : int) -> ndarray: ...
    def __surf(self, amount : int, minDistance : int) -> ndarray: ...