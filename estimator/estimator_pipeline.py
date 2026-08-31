
from __future__ import annotations

import numpy as np

from typing import Final
from numpy import ndarray
from abc import abstractmethod, ABC


class Estimator(ABC):

    def __init__(self): ...

    @abstractmethod
    def __next__(self, *args, **kwargs) -> None: ...

    @abstractmethod
    def __iter__(self) -> Estimator: ...


class Evaluator:

    LOSS_METRIX_SAMPSON : Final = "sampson_distance"
    LOSS_METRIX_PROJECTION : Final = "projection"

    def __init__(self):
        pass

    @staticmethod
    def sampson_distance(model: ndarray, features: ndarray) -> ndarray:
        """
        Sampson-Distance for each correspondence in px.

        F: (3, 3)
        p, q: (N, 3) homogene Punkte, Bild 1 bzw. Bild 2
        """

        P = features[..., 0]
        Q = features[..., 1]

        Fp = (model @ P.T).T
        Ftq = (model.T @ Q.T).T

        numerator = np.einsum('ij, ji -> i', Q, model @ P.T) ** 2
        denominator = Fp[:, 0] ** 2 + Fp[:, 1] ** 2 + Ftq[:, 0] ** 2 + Ftq[:, 1] ** 2

        eps = 1e-12
        safe_denom = np.where(denominator < eps, eps, denominator)
        return np.sqrt(numerator / safe_denom)

    @staticmethod
    def reprojection(F: ndarray, p: ndarray, q: ndarray) -> ndarray:
        """
        Symmetrische epipolare Distanz pro Korrespondenz, in Pixel - teurer
        als Sampson, aber naeher am tatsaechlichen geometrischen Fehler.
        Nuetzlich als Zweitmeinung bei starkem Rauschen/vielen Outliern.
        """
        Fp = (F @ p.T).T
        Ftq = (F.T @ q.T).T
        algebraic_loss = np.einsum('ij, ji -> i', q, F @ p.T)

        eps = 1e-12
        line2_norm = np.sqrt(Fp[:, 0] ** 2 + Fp[:, 1] ** 2)
        line1_norm = np.sqrt(Ftq[:, 0] ** 2 + Ftq[:, 1] ** 2)
        line2_norm = np.where(line2_norm < eps, eps, line2_norm)
        line1_norm = np.where(line1_norm < eps, eps, line1_norm)

        return np.abs(algebraic_loss) / line2_norm + np.abs(algebraic_loss) / line1_norm

    @staticmethod
    def l2_loss(F_ground: ndarray, F_res: ndarray) -> float:
        """Skalen-/vorzeichenkorrigierter L2-Fehler zwischen zwei F. Bereich [0, sqrt(2)]."""
        F_ground_n = F_ground / np.linalg.norm(F_ground)
        F_res_n = F_res / np.linalg.norm(F_res)

        positive_loss = np.linalg.norm(F_ground_n - F_res_n)
        negative_loss = np.linalg.norm(-F_ground_n - F_res_n)
        return float(min(positive_loss, negative_loss))
