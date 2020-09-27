from jax import numpy as np

class rustle(object):
    def __init__(
        self,
        n_tips_about_cowgirls_in_danger=20.,
        cowgirl_needs_rescuing_prob=.8,
        cowgirl_rescued_prob=.3,
        gratitude_mean=25.,
        gratitude_std=5.,
        n_cowboys=np.int64(10000),
        
        horses_in_the_front= None,
        horses_in_the_back = 2,
        khakhi_cowboy_hats = None,
        matte_black_hats = 1,
        
        # they glow in the dark, you just run them over
        CIA_agents = 213
    ):
        
        self.n_tips_about_cowgirls_in_danger = n_tips_about_cowgirls_in_danger
        self.cowgirl_needs_rescuing_prob = cowgirl_needs_rescuing_prob
        self.cowgirl_rescued_prob = cowgirl_rescued_prob
        self.gratitude_mean = gratitude_mean
        self.gratitude_std = gratitude_std
        self.n_cowboys = n_cowboys

