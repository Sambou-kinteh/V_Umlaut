import numpy as np

from MyHelpers.Frame import Frame
from estimator.ransac import Ransac
from feature_extraction.features import Features
from models.v_umlaut_solver import VUmlaut
from os import path


if __name__ == "__main__":

    path_ = "data/in/real/squirrel_images_and_data"
    img_names = ["image_34.jpg", "image_33.jpg"]
    frame1 = Frame(path.join(path_, img_names[0]), "frame1")
    frame2 = Frame(path.join(path_, img_names[1]), "frame2")

    N : int = 10000                          # number of iterations
    num_of_correspondences : int = 10000     # number of correspondences
    projection_threshold : float = 1e-6

    features: Features = Features(5, num_of_correspondences, frame1, frame2, 10, Features.EXTRACTOR_SIFT)
    model = VUmlaut(VUmlaut.METHOD_MATRIX)
    ransac = Ransac(model, None, features, projection_threshold)

    try:

        for i in range(N):
            if ransac.best_inlier_count == features.features.shape[0]: break
            print("iter: ", i)
            next(ransac)

    except StopIteration as e: print(e)

    finally:
        print("Correspondences: ", features.features.shape[0])
        print("Best Inlier Count: ", ransac.best_inlier_count)
        features.remove_outliers(ransac.best_inlier_mask)       # removes outliers
        print("\nbefore non-minimal refit: \n", np.round(model.model, 3))
        model.non_minimal_refit(np.vstack((ransac.best_point_configuration, features.features)).transpose(1, 0, 2))              # non-minimal refitting, todo remove best config points from features
        print("\nafter non-minimal refit: \n", np.round(model.model, 3))

