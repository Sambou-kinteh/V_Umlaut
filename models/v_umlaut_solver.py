
import numpy as np
import sympy as sp

from numpy import ndarray
from typing import Final
from model import Model


class VUmlaut(Model):

    METHOD_SYMBOLIC : Final = "symbolic_method"
    METHOD_MATRIX : Final = "matrix_method"

    # TODO: better to recieve the points as a matrix 5x2
    # TODO: use case: same instance used indefinitely

    def __init__(self, min_distance : int, method: str = METHOD_MATRIX):

        self.min_distance = min_distance
        self.method = method
        super().__init__()

    def isVUmlaut(self) -> bool:
        # check geometry , lines and geometry
        ...

    def determine_dependent_points(self) -> list[tuple]: ...

    def estimate_dependent_points(self): ...

    def perspective_normalisation(self): ...


    @staticmethod
    def symbolic_method(Z : ndarray, W : ndarray, H1 : ndarray, H2 : ndarray) -> ndarray:

        # MIT SYMPY IMPLEMENTIERT

        f12, f13, f21, f23, f31 = sp.symbols("f12 f13 f21 f23 f31")
        F : ndarray = np.array([
            [0, f12, f13],
            [f21, 0, f23],
            [f31, 1, 0]
        ])

        # gleichungssystem mit Gleichung 1
        system : list = [sp.expand(F.sum())]

        # Gleichungen 2, 3, 4
        for i in range(3):  # TODO erweitern für non minimal fitting

            system.append(sp.expand(
                (W[:, i].T @ F @ Z[:, i]).item()     # todo falls nicht dann einfache *
                # sp.Matrix.mutiply(sp.Matrix.multiply(W[:, i].T, F), Z[:, i])
            ))
        system.append(sp.expand(
            ((W[:, -1][-1] * Z[:, -1][0] * f23) / (W[:, -1][0] * Z[:, -1][-1])
             + (W[:, -2][0] * Z[:, -2][-2]) / (W[:, -2][-2] * Z[:, -2][0]))
        ))      # Gleichung 5, det constraint und wegwerfen von spurious fälle

        solution = sp.linsolve(system, [f12, f13, f21, f23, f31])
        assert solution is not sp.EmptySet, "No Solution recieve through symbolic method"
        assert len(solution) == 1, "Solution from symbolic method non linear"

        F[0, 1], F[0, 2], F[1, 0], F[1, 2], F[2, 0] = [float(each) for each in tuple(solution)[0]]
        F.astype(np.float64)        # safety measures

        return H2.T @ F @ H1        # F denormalisiert


    @staticmethod
    def matrix_method(Z : ndarray, W : ndarray, H1 : ndarray, H2 : ndarray) -> ndarray:

        # should get Z, W for i = 5, 6, 7
        # A und b aufstellen, A 5x5
        # f = A_inv * b

        # p4, q4 => z4, w4 => (1, 1, 1) führt dazu dass die einträge von F_coords zu 0 addieren
        # => es gibt eine 1er Spalte in A
        A : ndarray = np.zeros((5, 5), dtype=np.float64)
        b : ndarray = np.zeros((5,), dtype=np.float64)
        A[0, :] = 1
        b[0] = 0

        for i in range(3):      # todo: erweitern für non minimal fitting

            zi = Z[:, i]
            wi = W[:, i]
            A[i + 1, ...] = (wi[1]*zi[2], wi[1]*zi[3], wi[2]*zi[1], wi[2]*zi[3], wi[3]*zi[1])
            b[i + 1] = -wi[3]*zi[2]

        # det constraint für rg = 2 und linearität
        A[-1, 3] = (W[:, -1][-1] * Z[:, -1][0]) / (W[:, -1][0] * Z[:, -1][-1])      # w73*z71 / w71*z73
        b[-1] = - (W[:, -2][0] * Z[:, -2][-2]) / (W[:, -2][-2] * Z[:, -2][0])       # w61*z62 / w62*z61

        f = np.linalg.inv(A) @ b        # (f12, f13, f21, f23, f31)
        F : ndarray = np.array([
            [0, f[0], f[1]],
            [f[2], 0, f[3]],
            [f[4], 1, 0]
        ])
        return H2.T @ F @ H1        # F denormalisiert


    def __call__(self, *args, **kwargs) -> ndarray:

            """
            Must have a point argument
            :param args:
            :param kwargs:
            :return:
            """
            # recieves the 5 points and outputs F

            points : ndarray = kwargs.get("points")
            assert points is not None
            assert points.shape == (3, 5, 2)        # 2 dim matrix , (..., 0) -> p, (..., 1) -> q. punkte stacked vertikal
            assert self.isVUmlaut(points), "invalid points"

            A1_inv = np.linalg.inv(points[:, :3, 0])
            A2_inv = np.linalg.inv(points[:, :3, 1])

            D1_inv = np.diag(1 / (A1_inv @ points[:, 3, 0]))
            D2_inv = np.diag(1 / (A2_inv @ points[:, 3, 1]))

            H1 = D1_inv @ A1_inv
            H2 = D2_inv @ A2_inv

            Z = H1 @ points[:, 4, 0]   # i = 5
            W = H2 @ points[:, 4, 1]   # i = 5

            p6, q6, p7, q7 = self.determine_dependent_points()

            Z = np.column_stack((Z, H1 @ p6, H1 @ p7))  # i = 5, 6, 7
            W = np.column_stack((W, H2 @ q6, H2 @ q7))  # i = 5, 6, 7

            if self.method == self.METHOD_MATRIX: return self.matrix_method(Z, W, H1, H2)
            elif self.method == self.METHOD_SYMBOLIC: return self.symbolic_method(Z, W, H1, H2)

            return None

            # s11 = z6[0] / (z6[0] + z6[1])
            # s21 = w6[0] / (w6[0] + w6[1])
            #
            # s12 = z6[0] / (z6[0] + z6[2])
            # s22 = w6[0] / (w6[0] + w6[2])









if __name__ == "__main__":
    pass