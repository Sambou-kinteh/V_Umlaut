
import numpy as np
import sympy as sp

from numpy import ndarray
from typing import Final
from models.model import Model


class VUmlaut(Model):

    METHOD_SYMBOLIC : Final = "symbolic_method"
    METHOD_MATRIX : Final = "matrix_method"

    # todo convert asserts into raises and printouts after testing

    def __init__(self, method: str = METHOD_MATRIX):

        self.method = method
        self.__F = None
        super().__init__()

    @property
    def model(self): return self.__F

    @model.setter
    def model(self, model): self.__F = model

    def isVUmlaut(self, points : ndarray) -> bool:
        # check geometry , lines and geometry if neccessary
        ...

    def fit(self, points : ndarray, isMinimal : bool = True):

        # recieves the 5 points and sets model.model to newly determined F

        assert points is not None
        if isMinimal: assert points.shape == (3, 5, 2)        # 2 dim matrix , (..., 0) -> p, (..., 1) -> q. points stacked horizontally
        assert self.isVUmlaut(points), "Invalid configuration of points"

        #---------- perspective normalisation
        A1_inv = np.linalg.inv(points[:, :3, 0])
        A2_inv = np.linalg.inv(points[:, :3, 1])

        D1_inv = np.diag(1 / (A1_inv @ points[:, 3, 0]))
        D2_inv = np.diag(1 / (A2_inv @ points[:, 3, 1]))

        H1 = D1_inv @ A1_inv
        H2 = D2_inv @ A2_inv

        p6, q6, p7, q7 = self.determine_dependent_points()

        Z = H1 @ np.column_stack((points[:, 4, 0], p6, p7))   # i = 5, 6, 7
        W = H2 @ np.column_stack((points[:, 4, 1], q6, q7))   # i = 5, 6, 7

        if not isMinimal:
            Z_extra, W_extra = points[:, 5:, 0], points[:, 5:, 1]         # i = 6, ..., n
        else:
            Z_extra, W_extra = None, None

        #---------- solving and denormalising F
        if self.method == self.METHOD_MATRIX: self.model = H2.T @ self.matrix_method(Z, W, Z_extra, W_extra) @ H1
        elif self.method == self.METHOD_SYMBOLIC: self.model = H2.T @ self.symbolic_method(Z, W, Z_extra, W_extra) @ H1


    def project(self, features : ndarray):

        # features.shape = (N, 3, 2), W * F * Z.T
        return np.einsum('ij, ji -> i', features[..., 1], self.model @ features[..., 0].T)    # projection error

    def non_minimal_refit(self, points : ndarray): self.fit(points, False)

    def determine_dependent_points(self) -> list[tuple]: ...    # TODO TO IMPLEMENT

    @staticmethod
    def symbolic_method(Z : ndarray, W : ndarray, Z_extra : ndarray|None, W_extra : ndarray|None):

        #---------- implemented with sympy

        f12, f13, f21, f23, f31 = sp.symbols("f12 f13 f21 f23 f31")
        F : ndarray = np.array([
            [0, f12, f13],
            [f21, 0, f23],
            [f31, 1, 0]
        ])

        #---------- gleichungssystem
        # Gleichung 1
        system : list = [sp.expand(F.sum())]

        # Gleichungen 2, 3, 4 + (..., n)

        extra_dim = Z_extra.shape[1] if Z_extra is not None else 0
        for i in range(3 + extra_dim):

            system.append(sp.expand(
                (W[:, i].T @ F @ Z[:, i]).item()
                if i < 3
                else (W_extra[:, i - 3].T @ F @ Z_extra[:, i - 3]).item()

            ))  # sp.Matrix.mutiply(sp.Matrix.multiply(W[:, i].T, F), Z[:, i])

        # Gleichung 5, det constraint und wegwerfen von spurious fällen
        system.append(sp.expand(
            ((W[:, -1][-1] * Z[:, -1][0] * f23) / (W[:, -1][0] * Z[:, -1][-1])
             + (W[:, -2][0] * Z[:, -2][-2]) / (W[:, -2][-2] * Z[:, -2][0]))
        ))

        #---------- solution of system
        solution = sp.linsolve(system, [f12, f13, f21, f23, f31])
        assert solution is not sp.EmptySet, "No Solution recieve through symbolic method"
        assert len(solution) == 1, "Solution from symbolic method non linear"

        #---------- recovery of F
        F[0, 1], F[0, 2], F[1, 0], F[1, 2], F[2, 0] = [float(each) for each in tuple(solution)[0]]
        F.astype(np.float64)        # safety measure

        return F


    @staticmethod
    def matrix_method(Z : ndarray, W : ndarray, Z_extra : ndarray|None, W_extra : ndarray|None):

        # Z, W für i = 5, 6, 7
        # A und b aufstellen, A 5x5
        # f = A_inv * b

        # p4, q4 => z4, w4 => (1, 1, 1) führt dazu dass die einträge von F_coords zu 0 addieren
        # => es gibt eine 1er Spalte in A

        #---------- filling up 5x5 Matrix
        extra_dim = Z_extra.shape[1] if Z_extra is not None else 0
        A : ndarray = np.zeros((5 + extra_dim, 5), dtype=np.float64)
        b : ndarray = np.zeros((5 + extra_dim,), dtype=np.float64)
        A[0, :] = 1
        b[0] = -1

        for i in range(3):

            zi = Z[:, i]
            wi = W[:, i]
            A[i + 1, ...] = (wi[0]*zi[1], wi[0]*zi[2], wi[1]*zi[0], wi[1]*zi[2], wi[2]*zi[0])
            b[i + 1] = -wi[2]*zi[1]

        for i in range(extra_dim):

            zi = Z_extra[:, i]
            wi = W_extra[:, i]
            A[i + 4, ...] = (wi[0]*zi[1], wi[0]*zi[2], wi[1]*zi[0], wi[1]*zi[2], wi[2]*zi[0])
            b[i + 4] = -wi[2]*zi[1]

        # det constraint für rg = 2 und linearität
        A[-1, 3] = (W[:, -1][-1] * Z[:, -1][0]) / (W[:, -1][0] * Z[:, -1][-1])      # w73*z71 / w71*z73
        b[-1] = - (W[:, -2][0] * Z[:, -2][-2]) / (W[:, -2][-2] * Z[:, -2][0])       # w61*z62 / w62*z61

        #---------- linear solution for (f12, f13, f21, f23, f31)
        f = np.linalg.lstsq(A, b, rcond=None)[0]       # A hat maximal 5 von null verschiedenen Singulärwerte
        F : ndarray = np.array([
            [0, f[0], f[1]],
            [f[2], 0, f[3]],
            [f[4], 1, 0]
        ])

        return F


