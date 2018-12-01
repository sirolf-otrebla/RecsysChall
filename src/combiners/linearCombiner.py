

class linearCombiner():

    def combine(self, parameters, at=10):

        total_scores = 0
        for score in parameters:
            R, w = score[0], score[1]
            total_scores += w*R

        ranking = total_scores.argsort()[::-1]

        return ranking[:at].ravel()
