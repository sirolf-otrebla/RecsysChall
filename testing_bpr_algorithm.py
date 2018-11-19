from SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from refactored_code.utils import load_random_urms, load_icm


if __name__ == '__main__':
    urm_train, urm_test = load_random_urms()
    icm = load_icm()
    slim = SLIM_BPR_Cython(urm_train, positive_threshold=0.000005)
    slim.fit(epochs=3, validate_every_N_epochs=4, URM_test=urm_test, batch_size=1, sgd_mode='rmsprop', learning_rate=1e-4)
    print(slim.evaluateRecommendations(urm_train, urm_test, at=10))
