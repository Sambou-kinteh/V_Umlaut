
from MyHelpers.Frame import Frame
from estimator.ransac import Ransac
from feature_extraction.features import Features
from models.v_umlaut_solver import VUmlaut
from os import path


if __name__ == "__main__":

    path_ = "data/in/real/squirrel_images_and_data"
    img_names = ["image_3.jpg", "image_33.jpg"]
    frame1 = Frame(path.join(path_, img_names[0]), "frame1")
    frame2 = Frame(path.join(path_, img_names[1]), "frame2")

    N : int = 1000                          # number of iterations
    num_of_correspondences : int = 1000     # number of correspondences

    features: Features = Features(5, num_of_correspondences, frame1, frame2, 10, Features.EXTRACTOR_SIFT)
    model = VUmlaut(VUmlaut.METHOD_MATRIX)
    ransac = Ransac(model, None, features)

    try:

        for i in range(N): next(ransac)

    except StopIteration: pass

    finally:
        model.non_minimal_refit(features.features)  # non-minimal refitting

    print(model.model)