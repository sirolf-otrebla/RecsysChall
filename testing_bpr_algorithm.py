from Base.Evaluation.Evaluator import SequentialEvaluator
from SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from refactored_code.utils import load_random_urms, load_icm


if __name__ == '__main__':
    urm_train, urm_test = load_random_urms()
    icm = load_icm(0.7, 0.3, 0.5)
    slim =  SLIM_BPR_Cython(urm_train, positive_threshold=0, symmetric=True).fit(epochs=10,
                                                                                            topK=300,
                                                                                            batch_size=500,
                                                                                            sgd_mode='adagrad',
                                                                                            learning_rate=1e-4)
    evaluator_MF = SequentialEvaluator(urm_test, cutoff_list=[10])
    print(evaluator_MF.evaluateRecommender(slim))
