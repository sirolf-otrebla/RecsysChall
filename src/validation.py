import numpy as np

class Validator(object):

    _cumulative_precision = 0.0
    _cumulative_recall = 0.0
    _cumulative_MAP = 0.0

    _num_eval = 0

    def MAP(self, recommended_items, relevant_items):
        is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)

        # Cumulative sum: precision at 1, at 2, at 3 ...
        p_at_k = is_relevant * np.cumsum(is_relevant, dtype=np.float32) / (1 + np.arange(is_relevant.shape[0]))

        map_score = np.sum(p_at_k) / np.min([relevant_items.shape[0], is_relevant.shape[0]])

        return map_score

    def recall(self, recommended_items, relevant_items):
        is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)

        recall_score = np.sum(is_relevant, dtype=np.float32) / relevant_items.shape[0]

        return recall_score

    def precision(self, recommended_items, relevant_items):
        is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)

        precision_score = np.sum(is_relevant, dtype=np.float32) / len(is_relevant)

        return precision_score


    def evaluate_algorithm(self, userList_unique, URM_test, recommender_object, at=5):

        for user_id in userList_unique:

            relevant_items = URM_test[user_id].indices

            if len(relevant_items) > 0:
                recommended_items = recommender_object.recommend(user_id, at=at)
                self._num_eval += 1

                self._cumulative_precision += self.precision(recommended_items, relevant_items)
                self._cumulative_recall += self.recall(recommended_items, relevant_items)
                self._cumulative_MAP += self.MAP(recommended_items, relevant_items)

        self._cumulative_precision /= self._num_eval
        self._cumulative_recall /= self._num_eval
        self._cumulative_MAP /= self._num_eval

        print("Recommender performance is: Precision = {:.4f}, Recall = {:.4f}, MAP = {:.4f}".format(
            self._cumulative_precision, self._cumulative_recall, self._cumulative_MAP))