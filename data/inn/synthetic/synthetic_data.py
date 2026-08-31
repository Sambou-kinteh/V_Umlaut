
"""
Synthetische Szenengenerierung für Fundamental-Matrix Ground-Truth Tests.

Weltkoordinatensystem (ENU):
    +X = East, +Y = North, +Z = Up
Kamerakoordinatensystem (OpenCV):
    +X = Bild rechts, +Y = Bild runter, +Z = vorwaerts
Extrinsik-Konvention:
    X_cam = R @ X_world + t,   mit t = -R * c
wobei c das Kamerazentrum im Weltkoordinatensystem ist.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto

import numpy as np
from numpy import ndarray


class Mode(Enum):
    WAMI = auto()
    GENERAL = auto()


@dataclass
class SceneConfig:
    K: ndarray
    box_min: ndarray
    box_max: ndarray
    scene_center: ndarray
    cylinder_radius: float
    cam_z_low: float
    cam_z_high: float
    image_width: int
    image_height: int


@dataclass
class SceneData:
    """Kameraparameter + Korrespondenzen einer synthetischen Szene."""
    K1: ndarray = None
    K2: ndarray = None
    R1: ndarray = None
    R2: ndarray = None
    t1: ndarray = None
    t2: ndarray = None
    world_points: list = field(default_factory=list)
    Ps: ndarray = None  # (N, 3) homogen, Kamera 1
    Qs: ndarray = None  # (N, 3) homogen, Kamera 2

    def fundamental_matrix_ground_truth(self) -> ndarray:
        """
        Exakte, normierte Ground-Truth-F aus den bekannten Extrinsik-/
        Intrinsik-Parametern (fuer Matrix-Winkel-/Sampson-Fehler-Vergleiche).
        """
        R = self.R2 @ self.R1.T
        t = self.t2 - R @ self.t1

        t_cross = np.array([
            [0.0, -t[2], t[1]],
            [t[2], 0.0, -t[0]],
            [-t[1], t[0], 0.0],
        ])

        E = t_cross @ R
        F = np.linalg.inv(self.K2).T @ E @ np.linalg.inv(self.K1)
        return F / np.linalg.norm(F)

    def to_features(self) -> ndarray:
        """
        (N, 3, 2)

        """
        return np.stack([self.Ps, self.Qs], axis=-1)


def _scene_config(mode: Mode) -> SceneConfig:
    if mode == Mode.WAMI:
        # WAMI: lange Brennweite, flache Bodenszene, Kameras hoch ueber der
        # Szene innerhalb eines Zylinders.
        K = np.array([
            [17000.0, 0.0, 3300.0],
            [0.0, 17000.0, 2200.0],
            [0.0, 0.0, 1.0],
        ])
        return SceneConfig(
            K=K,
            box_min=np.array([-1000.0, -1000.0, 0.0]),
            box_max=np.array([1000.0, 1000.0, 100.0]),
            scene_center=np.array([0.0, 0.0, 50.0]),
            cylinder_radius=1500.0,
            cam_z_low=1500.0,
            cam_z_high=2000.0,
            image_width=6600,
            image_height=4400,
        )
    else:
        # General: VGA-Intrinsics, kleine Szene, Kameras innerhalb eines
        # umschliessenden Zylinders.
        K = np.array([
            [800.0, 0.0, 320.0],
            [0.0, 800.0, 240.0],
            [0.0, 0.0, 1.0],
        ])
        return SceneConfig(
            K=K,
            box_min=np.array([-2.0, -2.0, 0.0]),
            box_max=np.array([2.0, 2.0, 4.0]),
            scene_center=np.array([0.0, 0.0, 2.0]),
            cylinder_radius=8.0,
            cam_z_low=0.0,
            cam_z_high=8.0,
            image_width=640,
            image_height=480,
        )


def _look_at(from_: ndarray, to: ndarray) -> ndarray:
    """
    Baut die Welt-zu-Kamera-Rotationsmatrix R, deren Zeilen die
    Kamera-Achsen (right, cam_y, forward) in Weltkoordinaten sind.
    """
    forward = to - from_
    forward = forward / np.linalg.norm(forward)

    world_up = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(forward, world_up)) > 0.99:
        world_up = np.array([0.0, 1.0, 0.0])

    right = np.cross(forward, world_up)
    right = right / np.linalg.norm(right)

    cam_y = np.cross(forward, right)
    cam_y = cam_y / np.linalg.norm(cam_y)

    return np.vstack([right, cam_y, forward])


def _project_point(X: ndarray, K: ndarray, R: ndarray, t: ndarray) -> ndarray:
    """Projiziert einen 3D-Weltpunkt in homogene Pixelkoordinaten (x, y, 1)."""
    Xc = R @ X + t
    p = K @ Xc
    return p / p[2]


def _is_visible(X: ndarray, K: ndarray, R: ndarray, t: ndarray,
                width: int, height: int) -> bool:
    Xc = R @ X + t
    if Xc[2] <= 0.0:
        return False
    p = _project_point(X, K, R, t)
    return 0 <= p[0] < width and 0 <= p[1] < height


def _add_pixel_noise(data: SceneData, sigma: float, rng: np.random.Generator) -> None:
    if sigma <= 0.0:
        return
    data.Ps[:, :2] += rng.normal(0.0, sigma, size=(data.Ps.shape[0], 2))
    data.Qs[:, :2] += rng.normal(0.0, sigma, size=(data.Qs.shape[0], 2))


def inject_outliers(data: SceneData, n_outliers: int, rng: np.random.Generator) -> SceneData:

    n = data.Ps.shape[0]
    if n_outliers <= 0:
        return data
    if n_outliers > n:
        raise ValueError(f"n_outliers ({n_outliers}) > Anzahl Punkte ({n})")

    width2 = int(data.K2[0, 2] * 2)
    height2 = int(data.K2[1, 2] * 2)

    idx = rng.choice(n, size=n_outliers, replace=False)
    data.Qs[idx, 0] = rng.uniform(0, width2, size=n_outliers)
    data.Qs[idx, 1] = rng.uniform(0, height2, size=n_outliers)
    data.Qs[idx, 2] = 1.0
    return data


def generate(rng: np.random.Generator, mode: Mode = Mode.GENERAL,
             n_points: int = 5, independent_noise_sigma: float = 0.0) -> SceneData:
    """
    Generiert eine synthetische Szene mit n_points Weltpunkten, projiziert in
    zwei Kameras. Ps/Qs sind (n_points, 3) homogene Pixelkoordinaten.

    """
    cfg = _scene_config(mode)
    K = cfg.K

    data = SceneData()
    data.K1 = K.copy()
    data.K2 = K.copy()

    # Zwei Kameras gleichverteilt innerhalb eines massiven Zylinders sampeln.
    r1 = cfg.cylinder_radius * np.sqrt(rng.uniform(0.0, 1.0))
    r2 = cfg.cylinder_radius * np.sqrt(rng.uniform(0.0, 1.0))
    theta1 = rng.uniform(0.0, 2.0 * np.pi)
    theta2 = rng.uniform(0.0, 2.0 * np.pi)

    C1 = np.array([r1 * np.cos(theta1), r1 * np.sin(theta1),
                   rng.uniform(cfg.cam_z_low, cfg.cam_z_high)])
    C2 = np.array([r2 * np.cos(theta2), r2 * np.sin(theta2),
                   rng.uniform(cfg.cam_z_low, cfg.cam_z_high)])

    R1 = _look_at(C1, cfg.scene_center)
    R2 = _look_at(C2, cfg.scene_center)
    t1 = -R1 @ C1
    t2 = -R2 @ C2

    print(R1)
    print(R2)
    print(t1)
    print(t2)

    data.R1, data.R2 = R1, R2
    data.t1, data.t2 = t1, t2

    # n_points Punkte sampeln, die in beiden Kameras sichtbar sind.
    world_points = []
    for _ in range(n_points):
        for attempt in range(1, 100_001):
            X = np.array([
                rng.uniform(cfg.box_min[0], cfg.box_max[0]),
                rng.uniform(cfg.box_min[1], cfg.box_max[1]),
                rng.uniform(cfg.box_min[2], cfg.box_max[2]),
            ])
            if (_is_visible(X, K, R1, t1, cfg.image_width, cfg.image_height) and
                    _is_visible(X, K, R2, t2, cfg.image_width, cfg.image_height)):
                world_points.append(X)
                break
        else:
            raise RuntimeError(
                "Konnte nach 100000 Versuchen keinen sichtbaren Punkt finden."
            )
    data.world_points = world_points

    data.Ps = np.array([_project_point(X, K, R1, t1) for X in world_points])
    data.Qs = np.array([_project_point(X, K, R2, t2) for X in world_points])

    _add_pixel_noise(data, independent_noise_sigma, rng)

    return data


if __name__ == "__main__":
    ...
    # rng = np.random.default_rng(42)
    # scene = generate(rng, mode=Mode.GENERAL, n_points=10, independent_noise_sigma=0.0)
    # F_gt = scene.fundamental_matrix_ground_truth()
    # print(F_gt)
    #
    # residuals = [scene.Qs[i] @ F_gt @ scene.Ps[i] for i in range(scene.Ps.shape[0])]
    # print("Epipolar-Residuen (sollten ~0 sein):", residuals)
    #
    # vumlaut_points = scene.to_features()
    # print("VUmlaut-Format shape (erwartet (N, 3, 2)):", vumlaut_points.shape)
