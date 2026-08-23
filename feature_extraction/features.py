
import numpy as np
import cv2 as cv

from numpy import ndarray
from typing import Final
from MyHelpers.Frame import Frame


class Features:

    # TODO features werden bei der init definiert
    # TODO Features liefert alle punkte mit __next__() random i (hier 5) punkte ausspuckt
    # TODO speichert nach jeder iteration nur inliner
    # TODO kann am ende alle inliner ausgeben

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
        return self.__features

    @features.setter
    def features(self, features):

        assert self.__features is None, "Features have already been set"
        assert isinstance(features, ndarray), "Invalid feature type"
        self.__features = features

    def remove_outliners(self, outliners : ndarray):

        # todo should remove from both feature objects
        pass

    def __next__(self) -> ndarray:

        sample_row_indices = np.random.choice(self.__features.shape[0], size=self.__n, replace=False)   # replace = True for repeating points
        return self.__features[sample_row_indices, ...]     # (n, 3, 2)

    def __iter__(self): return self


    def __sift(self, N : int, minDistance : int, threshold : float = .75) -> ndarray:

        # TODO add min distance filtering

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

        good_matches = []
        for i, j in matches:
            if i.distance < threshold * j.distance:
                good_matches.append(i)

        points1 = np.float32([kp1[each.queryIdx].pt for each in good_matches])  # (N, 2)
        points2 = np.float32([kp2[each.trainIdx].pt for each in good_matches])  # (N, 2)

        points1_homogenous = np.column_stack([points1, np.ones(len(points1), dtype=np.float32)])  # (N, 3)
        points2_homogenous = np.column_stack([points2, np.ones(len(points2), dtype=np.float32)])  # (N, 3)

        return np.dstack([points1_homogenous, points2_homogenous])        # (N, 3, 2)


    def __orb(self, amount : int, minDistance : int) -> ndarray: ...
    def __surf(self, amount : int, minDistance : int) -> ndarray: ...