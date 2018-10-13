import scipy as sp

class Data_manager:

    def __init__(self, URM, ICM = 0):
        # if (URM is sp.sparse.coo_matrix ):
        self._cooURM = URM
        self._csrURM = URM.tocsr()
        self._cscURM = URM.tocsc()
        #else :
        #   i = 1
            # porcodio

    def getURM_CSR(self):
        return  self._csrURM

    def getURM_CSC(self):
        return  self._cscURM

    def getURM_COO(self):
        return  self._cooURM

