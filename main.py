
import cv2 as cv
import time

from MyHelpers.Frame import Frame
from estimator.ransac import Ransac
from feature_extraction.features import Features
from estimator.estimator_pipeline import Evaluator
from models.v_umlaut_solver import VUmlaut
from data.inn.synthetic.synthetic_data import *
from os import path


def opencv_methode(points: np.ndarray, method = cv.FM_RANSAC) -> tuple[list, ndarray]:
    # points.shape (3, N, 2)
    pts1 = points[:2, :, 0].T  # (N, 2) shape for OpenCV
    pts2 = points[:2, :, 1].T

    F, mask = cv.findFundamentalMat(pts1, pts2, method, ransacReprojThreshold=projection_threshold, maxIters=N, confidence=1)
    n_solutions = F.shape[0] // 3
    return [F[i*3:(i+1)*3] for i in range(n_solutions)], mask


#------------------------------ TEST RUN WITHOUT RANSAC (SYNTHETIC OR REAL) --------------------------------
def run_synthetic_without_ransac():

    round = 4
    print("\nRunning on synthetic data without ransac\n")
    print("Correspondences: ", features.features.shape[0])
    print("\nGround Truth:\n", np.round(features.ground_truth, round), "\n")

    if features.ground_truth is None: raise ValueError("No Synthetic Data defined for synthetic test")
    if features.features.shape[0] != 7: raise ValueError("Too many points defined for test")

    feature = features.features.transpose(1, 0, 2)
    start_time = time.time()
    model.fit(feature[:, :5])
    v_umlaut_time = time.time() - start_time

    print("\nV-Umlaut F: \n", np.round(model.model / np.linalg.norm(model.model), round))
    print("L2 Loss: ", np.round(Evaluator.l2_loss(features.ground_truth, model.model), round))
    print("V-Umlaut Time: ", np.round(v_umlaut_time*1000, 3), "ms")

    start_time = time.time()
    opencv_out : list = opencv_methode(feature, cv.FM_7POINT)[0]
    cv_time = time.time() - start_time

    print("\nOpenCV F: \n", np.round(np.array(opencv_out), round))
    print("L2 Loss: ", *list(map(lambda res: np.round(Evaluator.l2_loss(features.ground_truth, res), round), opencv_out)))
    print("OpenCV Time: ", np.round(cv_time*1000, 3), "ms")


#------------------------------ TEST RUN WITH RANSAC (SYNTHETIC OR REAL) --------------------------------
def run_with_ransac():

    round = 4
    isSynthetic = features.ground_truth is not None

    print(f"\nRunning on {"Synthetic" if isSynthetic else "Real"} Data with ransac\n")
    if isSynthetic: print("\nGround Truth:\n", np.round(features.ground_truth, round), "\n")
    print("\nCorrespondences: ", features.features.shape[0], "\n")

    start_time = time.time()
    try:

        for i in range(N):
            if ransac.best_inlier_count == features.features.shape[0]: break
            if ransac.iters_after_best >= N//7: break
            next(ransac)

    except StopIteration as e: print(e)

    finally:

        features_copy = features.features.copy().transpose(1, 0, 2)

        print("Best Inlier Count (V-Umlaut): ", ransac.best_inlier_count)
        print("\nV-Umlaut F: \n", np.round(model.model / np.linalg.norm(model.model), round))
        if isSynthetic: print("L2 Loss: ", np.round(Evaluator.l2_loss(features.ground_truth, model.model), round))
        print("V-Umlaut Time: ", np.round(time.time() - start_time, 3), "s")
        print("V-Umlaut S Parameters: ", model.v_umlaut_sij)

        features.remove_outliers(ransac.best_inlier_mask)       # removes outliers
        model.non_minimal_refit(np.vstack((ransac.best_point_confi, features.features)).transpose(1, 0, 2))

        print("\nV-Umlaut F (NM): \n", np.round(model.model / np.linalg.norm(model.model), round))
        if isSynthetic: print("L2 Loss: ", np.round(Evaluator.l2_loss(features.ground_truth, model.model), round))
        print("V-Umlaut Time (Total): ", np.round(time.time() - start_time, 3), "s")

        start_time = time.time()
        opencv_out, mask = opencv_methode(features_copy, cv.FM_RANSAC)
        cv_time = time.time() - start_time

        print("\n\nBest Inlier Count (OpenCV): ", mask.sum())
        print("\nOpenCV F: \n", np.round(np.array(opencv_out), round))
        if isSynthetic: print("L2 Loss: ", np.round(Evaluator.l2_loss(features.ground_truth, np.array(opencv_out)), round))
        print("OpenCV Time: ", np.round(cv_time, 3), "s")



#------------------------------ RUN --------------------------------
if __name__ == "__main__":

    path_ = "data/inn/real/squirrel_images_and_data"
    img_names = ["image_34.jpg", "image_33.jpg"]
    frame1 = Frame(path.join(path_, img_names[0]), "frame1")
    frame2 = Frame(path.join(path_, img_names[1]), "frame2")

    N : int = 10000                          # number of iterations
    num_of_correspondences : int = 10000     # number of correspondences
    percent_outlier = 40                     # 10%-40% z.B
    projection_threshold : float = 1e-2
    
    synthetic_params = {
        "seed" : None,
        "sigma" : 10,
        "outliers" : int(percent_outlier/100 * num_of_correspondences),
    }

    features: Features = Features(5, num_of_correspondences, frame1, frame2, Features.EXTRACTOR_SYNTHETIC, **synthetic_params)
    model = VUmlaut(VUmlaut.METHOD_MATRIX)
    ransac = Ransac(model, features, projection_threshold, useLO=True)

    run_with_ransac()


