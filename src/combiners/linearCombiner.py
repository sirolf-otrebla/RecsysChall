import numpy as np


class linearCombiner():

    def combine(self, parameters, at=10):

        total_scores = 0
        for score in parameters:
            R = score[0]
            w = score[1]
            total_scores += np.multiply(R, w)

        ranking = total_scores.argsort()[::-1]

        return ranking[:at]
