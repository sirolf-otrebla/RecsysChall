
import numpy as np


class ProbabilisticCombiner():

    def combine(self, scores, alpha=0.5, at=10):

        first_reclist = scores[0][0]
        second_reclist = scores[1][0]

        result = []
        i = 0
        while i < at:
            rand = np.random.uniform(0, 1)
            if rand < alpha:
                if type(first_reclist) is np.ndarray:
                    chosen = first_reclist[0]
                else:
                    chosen = first_reclist
                first_reclist = np.delete(first_reclist, 0)
            else:
                if type(second_reclist) is np.ndarray:
                    chosen = second_reclist[0]
                else:
                    chosen = second_reclist
                second_reclist = np.delete(second_reclist, 0)
            if chosen in result:
                continue
            else:
                result.append(chosen)
                i += 1


        return result
