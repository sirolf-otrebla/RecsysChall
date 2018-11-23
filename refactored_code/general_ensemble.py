from Base.Cython.cosine_similarity import Cosine_Similarity
from refactored_code.IALS_numpy import IALS_numpy
from SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from sklearn import preprocessing
import numpy as np


class GeneralEnsemble:

    def __init__(self, URM_train, URM_test, ICM, k=100,
                 alpha= 0.7662,
                 beta= 0.8188,
                 gamma=0.30,
                 epsilon=0.6212,
                 ro=0.80,
                 mu = 0.45,
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
        self.SLIM_BPRUU = mu

    def fit(self, alpha):
        self._similarity_matrixUU = Cosine_Similarity(self._URM_train.T,
                                                      self._k,
                                                      self._shrink,
                                                      normalize=True,
                                                      mode='cosine').compute_similarity()

        self._similarity_matrixII = Cosine_Similarity(self._URM_train.tocsc(),
                                                      self._k,
                                                      self._shrink,
                                                      normalize=True,
                                                      mode='cosine').compute_similarity()
        self._similarity_matrixCBF = Cosine_Similarity(self._ICM.T,
                                                       self._k,
                                                       self._shrink,
                                                       normalize=True,
                                                       mode='cosine').compute_similarity()
        self.latent_x, self.latent_y = (IALS_numpy(reg=alpha)).fit(self._URM_train)

        self.bpr_WII = SLIM_BPR_Cython(self._URM_train, positive_threshold=0, symmetric=False).fit(epochs=10,
                                                                                                   topK=100,
                                                                                                   validate_every_N_epochs=11,
                                                                                                   URM_test=self._URM_test,
                                                                                                    batch_size=500,
                                                                                                   sgd_mode='adagrad',
                                                                                                   learning_rate=1e-4)

        self.bpr_WUU = SLIM_BPR_Cython(self._URM_train.T, positive_threshold=0).fit(epochs=7,
                                                                                    validate_every_N_epochs=8,
                                                                                    URM_test=self._URM_test.T,
                                                                                    batch_size=500,
                                                                                    sgd_mode='adagrad',
                                                                                    learning_rate=1e-4)
        # self.bpr_WUU = preprocessing.normalize(self.bpr_WUU, norm='max', axis=0)
        # self.bpr_WII = preprocessing.normalize(self.bpr_WII, norm='max', axis=1)
        # self._similarity_matrixCBF = preprocessing.normalize(self._similarity_matrixCBF, norm='max', axis=1)
        # self._similarity_matrixII = preprocessing.normalize(self._similarity_matrixII, norm='max', axis=1)
        # self._similarity_matrixUU = preprocessing.normalize(self._similarity_matrixUU, norm='max', axis=1)

    def recommend(self, user_id, at=None, exclude_seen=True):
        # compute the scores using the dot product
        user_profile = self._URM_train[user_id]
        # normalized_IALS = np.dot(self.latent_x[user_id], self.latent_y.T)

        normalized_IALS = preprocessing.normalize(np.dot(self.latent_x[user_id], self.latent_y.T), axis=1, norm='max')
        cfii = self.IISCORE*preprocessing.normalize(user_profile.dot(self._similarity_matrixII).toarray(), norm='max')
        cfuu = self.UUSCORE*preprocessing.normalize(self._similarity_matrixUU[user_id].dot(self._URM_train).toarray(),
                                                    norm='max', axis=1)
        cbf =  self.CBFSCORE*preprocessing.normalize(self.CBFSCORE*user_profile.dot(self._similarity_matrixCBF).toarray(),
                                                     norm='max', axis=1)
        bprii = self.SLIM_BPR*preprocessing.normalize(user_profile.dot(self.bpr_WII.T).toarray(), norm='max')
        bpruu = self.SLIM_BPRUU*preprocessing.normalize(self.bpr_WUU[user_id].dot(self._URM_train).toarray(), norm='max')

        scores = (cfii + cfuu + cbf + bprii + normalized_IALS + bpruu).ravel()
        # scores = (self.IISCORE*user_profile.dot(self._similarity_matrixII).toarray() +
        #          self.UUSCORE*self._similarity_matrixUU[user_id].dot(self._URM_train).toarray() +
        #          self.CBFSCORE*user_profile.dot(self._similarity_matrixCBF).toarray() +
        #          self.IALSSCORE*normalized_IALS +
        #          self.SLIM_BPR*user_profile.dot(self.bpr_WII.T).toarray()
        #          #self.SLIM_BPR*self.bpr_WUU[user_id].dot(self._URM_train).toarray()
        #          ).ravel()

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
