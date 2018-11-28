from SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from refactored_code.utils import load_random_urms, load_icm


if __name__ == '__main__':
    urm_train, urm_test = load_random_urms()
    icm = load_icm()
    slim = SLIM_BPR_Cython(urm_train.T, positive_threshold=0)
    slim.fit(epochs=20, validate_every_N_epochs=20, URM_test=urm_test.T, batch_size=500, sgd_mode='adagrad', learning_rate=1e-4)
    print(slim.evaluateRecommendations(urm_train, urm_test, at=10))
