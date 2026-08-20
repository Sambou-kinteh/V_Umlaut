
import cv2 as cv
import numpy as np

from numpy import ndarray
from typing import final


class VUmlaut:

    METHOD_SYMBOLIC : final = "symbolic_method"
    METHOD_MATRIX : final = "matrix_method"

    # TODO: better to recieve the points as a matrix 5x2

    def __init__(self, min_distance : int, method: str):

        self.min_distance : int = min_distance
        ...

    def isVUmlaut(self):
        # check geometry , lines and geometry
        ...

    def estimate_dependent_points(self): ...

    def perspective_normalisation(self): ...


    def symbolic_method(self): ...

    def matrix_method(self): ...

    def __call__(self, *args, **kwargs):
        # recieves the 5 points and outputs F
        ...









if __name__ == "__main__":
    pass