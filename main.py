import numpy as np

from MyHelpers.Frame import Frame
from estimator.ransac import Ransac
from feature_extraction.features import Features
from models.v_umlaut_solver import VUmlaut
from os import path


if __name__ == "__main__":

    path_ = "data/in/real/squirrel_images_and_data"
    img_names = ["image_1.jpg", "image_35.jpg"]
    frame1 = Frame(path.join(path_, img_names[0]), "frame1")
    frame2 = Frame(path.join(path_, img_names[1]), "frame2")

    N : int = 1000000                          # number of iterations
    num_of_correspondences : int = 10000     # number of correspondences

    features: Features = Features(5, num_of_correspondences, frame1, frame2, 10, Features.EXTRACTOR_SIFT)
    print("Correspondences: ", features.features.shape[0])
    input()     # TODO
    model = VUmlaut(VUmlaut.METHOD_MATRIX)
    ransac = Ransac(model, None, features, .001)

    try:

        for i in range(N):
            if ransac.best_inlier_count == features.features.shape[0]: break
            print("iter: ", i)
            next(ransac)

    except StopIteration as e: print(e)

    finally:
        features.remove_outliers(ransac.best_inlier_mask)       # removes outliers
        print("before non-minimal refit: \n", model.model)
        model.non_minimal_refit(np.vstack((ransac.best_point_configuration, features.features)).transpose(1, 0, 2))              # non-minimal refitting, todo remove best config points from features
        print("before non-minimal refit: \n", model.model)

