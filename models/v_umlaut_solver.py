
import numpy as np
import sympy as sp

from numpy import ndarray
from typing import Final
from models.model import Model


class VUmlaut(Model):

    METHOD_SYMBOLIC : Final = "symbolic_method"
    METHOD_MATRIX : Final = "matrix_method"

    def __init__(self, method: str = METHOD_MATRIX):

        self.method = method
        self.__F = None
        self.v_umlaut_sij = 0.4971              # can be changed during LO in RANSAC, experimental average
        super().__init__()

        #----------------------- Experimental Values
        # current average = 0.4971
        self.good_values = [0.5110551259847875, 0.33914248024278837, 0.18233438184315487, 0.5967517839931419, 0.32076094673598354,
                            0.4235798829852677, 0.09844013399589535, 0.5500227949478695, 0.6207799337997877, 0.2816573430926693,
                            0.4726866883882649, 0.06757658867159988, 0.9322628217108037, 0.7199098545025675, 0.6115610118705175,
                            0.32853819062504275, 0.7719097486076748, 0.9116853972465421, 0.47849023820184067, 0.5616037344954625,
                            0.6590904757193937]

    @property
    def model(self): return self.__F

    @model.setter
    def model(self, model): self.__F = model

    @staticmethod
    def isVUmlaut(points : ndarray, corresp4) -> bool:

        delta : float = 1e-4

        if np.abs(np.linalg.det(points[..., 0])) < delta or np.abs(np.linalg.det(points[..., 1])) < delta: return False
        elif (np.any(np.abs(np.linalg.inv(points[..., 0]) @ corresp4[..., 0]) < delta) or
              np.any(np.abs(np.linalg.inv(points[..., 1]) @ corresp4[..., 1]) < delta)): return False

        return True

    def fit(self, points : ndarray, isMinimal : bool = True):

        # recieves the 5 points and sets model.model to newly determined F
        # expected (3, 5, 2)

        if points is None: raise ValueError("No points to fit to model")

        k = points.shape[1]
        if isMinimal and k != 5: raise ValueError("Invalid number of correspondences")
        if not self.isVUmlaut(points[:, :3], points[:, 3, :]): return

        #---------- perspective normalisation
        A1_inv = np.linalg.inv(points[:, :3, 0])
        A2_inv = np.linalg.inv(points[:, :3, 1])

        D1_inv = np.diag(1 / (A1_inv @ points[:, 3, 0]))
        D2_inv = np.diag(1 / (A2_inv @ points[:, 3, 1]))

        H1 = D1_inv @ A1_inv
        H2 = D2_inv @ A2_inv

        p6, q6, p7, q7 = self.determine_dependent_points(points[:, :3])

        Z = H1 @ np.column_stack((points[:, 4, 0], p6, p7))     # i = 5, 6, 7
        W = H2 @ np.column_stack((points[:, 4, 1], q6, q7))     # i = 5, 6, 7

        if isMinimal: Z_extra, W_extra = None, None
        else: Z_extra, W_extra = H1 @ points[:, 5:, 0], H2 @ points[:, 5:, 1]         # i = 6, ..., n

        #---------- argment points
        iter_over = (0, )
        for i in range(len(iter_over) + (0 if Z_extra is None else Z_extra.shape[1])):

            delta = 1e-6
            if i < len(iter_over):
                if abs(Z[:, i][-1]) > delta: Z[:, i] /= Z[:, i][-1]
                if abs(W[:, i][-1]) > delta: W[:, i] /= W[:, i][-1]

            else:
                i -= len(iter_over)
                if abs(Z_extra[:, i][-1]) > delta: Z_extra[:, i] /= Z_extra[:, i][-1]
                if abs(W_extra[:, i][-1]) > delta: W_extra[:, i] /= W_extra[:, i][-1]

        #---------- bringing i = 6, 7 into barycentric coordinates: !=0 and !=1

        barycentric_coord = lambda coord, other_index: (coord[0] / coord[other_index]) / (1 + coord[0] / coord[other_index])
        s11 = barycentric_coord(Z[:, 1], 1)
        s21 = barycentric_coord(W[:, 1], 1)

        s12 = barycentric_coord(Z[:, -1], -1)
        s22 = barycentric_coord(W[:, -1], -1)

        Z[:, 1] = (s11, 1-s11, 0)
        W[:, 1] = (s21, 1-s21, 0)

        Z[:, -1] = (s12, 0, 1-s12)
        W[:, -1] = (s22, 0, 1-s22)

        #---------- solving and denormalising F
        if self.method == self.METHOD_MATRIX: self.model = H2.T @ self.matrix_method(Z, W, Z_extra, W_extra) @ H1
        elif self.method == self.METHOD_SYMBOLIC: self.model = H2.T @ self.symbolic_method(Z, W, Z_extra, W_extra) @ H1

    def project(self, features : ndarray):

        # features.shape = (N, 3, 2), W * F * Z.T
        return np.einsum('ij, ji -> i', features[..., 1], self.model @ features[..., 0].T)    # projection error

    def non_minimal_refit(self, points : ndarray):

        assert points.shape[1] >= 5, "Not enough points to continue"
        self.fit(points, False)

    def determine_dependent_points(self, points : ndarray) -> tuple:

        # let
        s11, s21, s12, s22 = [self.v_umlaut_sij] * 4
        p1 = points[:, 0, 0]
        q1 = points[:, 0, 1]

        return (
            s11 * p1 + (1 - s11) * points[:, 1, 0],         # p6 = s11 * p1 + (1- s11) * p2
            s21 * q1 + (1 - s21) * points[:, 1, 1],         # q6 = s21 * q1 + (1- s21) * q2
            s12 * p1 + (1 - s12) * points[:, -1, 0],        # p7 = s12 * p1 + (1- s12) * p3
            s22 * q1 + (1 - s22) * points[:, -1, 1]         # q7 = s22 * q1 + (1- s22) * q3
        )

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
                (W[:, i].T @ F @ Z[:, i])
                if i < 3
                else (W_extra[:, i - 3].T @ F @ Z_extra[:, i - 3])

            ))

        # Gleichung 5, det constraint und wegwerfen von spurious fällen
        system.append(sp.expand(
            (
                    (W[:, -1][0] * Z[:, -1][-1] * W[:, -2][-2] * Z[:, -2][0] * f23)
                    +
                    (W[:, -2][0] * Z[:, -2][-2] * W[:, -1][-1] * Z[:, -1][0])
            )
        ))

        #---------- solution of system
        solution = sp.linsolve(system, [f12, f13, f21, f23, f31])
        assert solution is not sp.EmptySet, "No Solution recieve through symbolic method"
        assert len(solution) == 1, "Solution from symbolic method non linear"

        #---------- recovery of F
        F[0, 1], F[0, 2], F[1, 0], F[1, 2], F[2, 0] = [float(each) for each in tuple(solution)[0]]
        F.astype(np.float64)        # safety measure

        # for example in the paper
        # print("Symbolische Methode (mit Sympy):")
        # print(F)

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
        A[-1, 3] = W[:, -1][0] * Z[:, -1][-1] * W[:, -2][-2] * Z[:, -2][0]  # w71 * z73 * w62 * z61
        b[-1] = -W[:, -2][0] * Z[:, -2][-2] * W[:, -1][-1] * Z[:, -1][0]    # w61 * z62 * w73 * z71

        delta = 1e-9
        A[np.abs(A) < delta] = 0
        b[np.abs(b) < delta] = 0

        #---------- linear solution for (f12, f13, f21, f23, f31)
        f = np.linalg.lstsq(A, b, rcond=None)[0]       # A hat maximal 5 von null verschiedenen Singulärwerte
        F : ndarray = np.array([
            [0, f[0], f[1]],
            [f[2], 0, f[3]],
            [f[4], 1, 0]
        ])

        return F


