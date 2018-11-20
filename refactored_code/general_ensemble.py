from Base.Cython.cosine_similarity import Cosine_Similarity
from refactored_code.IALS_numpy import IALS_numpy
from SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from sklearn import preprocessing
import numpy as np


class GeneralEnsemble:

    def __init__(self, URM_train, URM_test, ICM, k=100,
                 alpha= 0, #0.7662,
                 beta= 0, #0.8188,
                 gamma=0, #0.3325,
                 epsilon=0 , #0.6212,
                 ro=1.5,
                 shrink=15):

        self._URM_train = URM_train.tocsr()
        self._URM_test = URM_test.tocsr()
        self._ICM = ICM.tocsr()
        self._k = k
        self._shrink = shrink

        self.UUSCORE = alpha

        self.IISCORE = beta

        self. CBFSCORE = gamma

        self.IALSSCORE = epsilon

        self.SLIM_BPR = ro

    def fit(self, alpha):
        self._similarity_matrixUU = Cosine_Similarity(self._URM_train.T,
                                                      self._k,
                                                      self._shrink,
                                                      normalize=True,
                                                      mode='cosine').compute_similarity()
        self._similarity_matrixUU = preprocessing.normalize(self._similarity_matrixUU, norm='max', axis=0)

        self._similarity_matrixII = Cosine_Similarity(self._URM_train.tocsc(),
                                                      self._k,
                                                      self._shrink,
                                                      normalize=True,
                                                      mode='cosine').compute_similarity()
        self._similarity_matrixII = preprocessing.normalize(self._similarity_matrixII, norm='max', axis=0)


        self._similarity_matrixCBF = Cosine_Similarity(self._ICM.T,
                                                       self._k,
                                                       self._shrink,
                                                       normalize=True,
                                                       mode='cosine').compute_similarity()
        self._similarity_matrixCBF = preprocessing.normalize(self._similarity_matrixCBF, norm='max', axis=0)

        # self.latent_x, self.latent_y = (IALS_numpy(reg=alpha)).fit(self._URM_train)

        self.bpr_WII = SLIM_BPR_Cython(self._URM_train, positive_threshold=0.6).fit(epochs=6, validate_every_N_epochs=1, URM_test=self._URM_test,
                                                                                                    batch_size=1, sgd_mode='sgd', learning_rate=1e-4)
        self.bpr_WII = preprocessing.normalize(self.bpr_WII, norm='max', axis=1)

        # self.bpr_WUU = SLIM_BPR_Cython(self._URM_train.T, positive_threshold=0.6).fit(epochs=7, validate_every_N_epochs=1, URM_test=self._URM_test.T,
        #                                                                                           batch_size=1, sgd_mode='rmsprop', learning_rate=1e-4)
        # self.bpr_WUU = preprocessing.normalize(self.bpr_WUU, norm='max', axis=0)


    def recommend(self, user_id, at=None, exclude_seen=True):
        # compute the scores using the dot product
        user_profile = self._URM_train[user_id]
        # normalized_IALS = preprocessing.normalize(np.dot(self.latent_x[user_id], self.latent_y.T), axis=0, norm='max')
        scores = (self.IISCORE*user_profile.dot(self._similarity_matrixII).toarray() +
                  self.UUSCORE*self._similarity_matrixUU[user_id].dot(self._URM_train).toarray() +
                  self.CBFSCORE*user_profile.dot(self._similarity_matrixCBF).toarray() +
                  # self.IALSSCORE*normalized_IALS +
                  self.SLIM_BPR*user_profile.dot(self.bpr_WII).toarray()
                  #self.SLIM_BPR*self.bpr_WUU[user_id].dot(self._URM_train).toarray()
                  ).ravel()

        if exclude_seen:
            scores = self.filter_seen(user_id, scores)

        # rank items
        ranking = scores.argsort()[::-1]

        return ranking[:at]

    def filter_seen(self, user_id, scores):

        start_pos = int(self._URM_train.indptr[user_id])
        end_pos = int(self._URM_train.indptr[user_id + 1])

        user_profile = self._URM_train.indices[start_pos:end_pos]

        scores[user_profile] = -np.inf

        return scores

    def recommendALL(self, userList, at=10):
        res = np.array([])
        n=0
        for i in userList:
            n+=1
            recList = self.recommend(i, at)
            tuple = np.concatenate((i, recList))
            if (res.size == 0):
                res = tuple
            else:
                res = np.vstack([res, tuple])
        return res
