from refactored_code.mauri_code.Base.Recommender import Recommender
from refactored_code.mauri_code.Base.Recommender_utils import check_matrix

class BPRMF(Recommender):

    '''

    BPRMF model

    '''



    # TODO: add global effects

    def __init__(self,

                 num_factors=50,

                 lrate=0.01,

                 user_reg=0.015,

                 pos_reg=0.015,

                 neg_reg=0.0015,

                 iters=10,

                 sampling_type='user_uniform_item_uniform',

                 sample_with_replacement=True,

                 use_resampling=True,

                 sampling_pop_alpha=1.0,

                 init_mean=0.0,

                 init_std=0.1,

                 lrate_decay=1.0,

                 rnd_seed=42,

                 verbose=True):

        '''

        Initialize the model

        :param num_factors: number of latent factors

        :param lrate: initial learning rate used in SGD

        :param user_reg: regularization for the user factors

        :param pos_reg: regularization for the factors of the positive sampled items

        :param neg_reg: regularization for the factors of the negative sampled items

        :param iters: number of iterations in training the model with SGD

        :param sampling_type: type of sampling. Supported types are 'user_uniform_item_uniform' and 'user_uniform_item_pop'

        :param sample_with_replacement: `True` to sample positive items with replacement (doesn't work with 'user_uniform_item_pop')

        :param use_resampling: `True` to resample at each iteration during training

        :param sampling_pop_alpha: float smoothing factor for popularity based samplers (e.g., 'user_uniform_item_pop')

        :param init_mean: mean used to initialize the latent factors

        :param init_std: standard deviation used to initialize the latent factors

        :param lrate_decay: learning rate decay

        :param rnd_seed: random seed

        :param verbose: controls verbosity in output

        '''

        super(BPRMF, self).__init__()

        self.num_factors = num_factors

        self.lrate = lrate

        self.user_reg = user_reg

        self.pos_reg = pos_reg

        self.neg_reg = neg_reg

        self.iters = iters

        self.sampling_type = sampling_type

        self.sample_with_replacement = sample_with_replacement

        self.use_resampling = use_resampling

        self.sampling_pop_alpha = sampling_pop_alpha

        self.init_mean = init_mean

        self.init_std = init_std

        self.lrate_decay = lrate_decay

        self.rnd_seed = rnd_seed

        self.verbose = verbose



    def __str__(self):

        return "BPRMF(num_factors={}, lrate={}, user_reg={}. pos_reg={}, neg_reg={}, iters={}, " \
               "sampling_type={}, sample_with_replacement={}, use_resampling={}, sampling_pop_alpha={}, init_mean={}, " \
               "init_std={}, lrate_decay={}, rnd_seed={}, verbose={})".format(

            self.num_factors, self.lrate, self.user_reg, self.pos_reg, self.neg_reg, self.iters,

            self.sampling_type, self.sample_with_replacement, self.use_resampling, self.sampling_pop_alpha,

            self.init_mean,

            self.init_std,

            self.lrate_decay,

            self.rnd_seed,

            self.verbose

        )



    def fit(self, R):

        self.dataset = R

        R = check_matrix(R, 'csr', dtype=np.float32)

        self.X, self.Y = BPRMF_sgd(R,

                                   num_factors=self.num_factors,

                                   lrate=self.lrate,

                                   user_reg=self.user_reg,

                                   pos_reg=self.pos_reg,

                                   neg_reg=self.neg_reg,

                                   iters=self.iters,

                                   sampling_type=self.sampling_type,

                                   sample_with_replacement=self.sample_with_replacement,

                                   use_resampling=self.use_resampling,

                                   sampling_pop_alpha=self.sampling_pop_alpha,

                                   init_mean=self.init_mean,

                                   init_std=self.init_std,

                                   lrate_decay=self.lrate_decay,

                                   rnd_seed=self.rnd_seed,

                                   verbose=self.verbose)



    def recommend(self, user_id, n=None, exclude_seen=True):

        scores = np.dot(self.X[user_id], self.Y.T)

        ranking = scores.argsort()[::-1]

        # rank items

        if exclude_seen:

            ranking = self._filter_seen(user_id, ranking)

        return ranking[:n]







    def _get_user_ratings(self, user_id):

        return self.dataset[user_id]



    def _get_item_ratings(self, item_id):

        return self.dataset[:, item_id]





    def _filter_seen(self, user_id, ranking):

        user_profile = self._get_user_ratings(user_id)

        seen = user_profile.indices

        unseen_mask = np.in1d(ranking, seen, assume_unique=True, invert=True)

        return ranking[unseen_mask]