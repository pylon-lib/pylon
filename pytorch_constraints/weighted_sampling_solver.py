import torch
import torch.random

import itertools
from random import Random

from .solver import Solver
from .utils import decoding_loss

from torch.distributions import *

class WeightedSamplingSolver(Solver):
    '''Sample from a multinomial distribution parameterized by the network's probabilities'''

    def __init__(self, num_samples=2):
        self.num_samples = num_samples
        self.random = Random(torch.random.get_rng_state())

    def sample(self, logits):
        '''Sample a decoding of the variables where each variable $X_i$ is sampled with $p_i$'''

        # We're only interested in p, and not 1-p
        probs = torch.softmax(logits, dim=-1)[:, -1]
        
        while(True):
            yield torch.bernoulli(probs).long() 

    def loss(self, logits):
        sampler = self.sample(logits)
        samples = [next(sampler) for _ in range(self.num_samples)]
        log_probs = torch.log_softmax(logits, dim=-1)

        satis_losses = tuple(map(lambda sample: decoding_loss(sample, log_probs),
                           filter(self.cond, samples)))
        satis_loss = torch.stack(satis_losses).logsumexp(dim=0)\
            if satis_losses else 0

        viol_losses = tuple(map(lambda sample: decoding_loss(sample, log_probs),
                          filter(lambda v: not self.cond(v), samples)))
        viol_loss = torch.stack(viol_losses).logsumexp(dim=0)\
            if viol_losses else 0

        return viol_loss - satis_loss
