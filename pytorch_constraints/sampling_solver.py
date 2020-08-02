import torch
import torch.random

import itertools
from random import Random

from .solver import Solver
from .utils import decoding_loss


class SamplingSolver(Solver):
    '''Samples possible decodings and defines the loss based on whether the constraint is satisfied or violated for each.'''

    def __init__(self, num_samples=2):
        self.num_samples = num_samples
        self.random = Random(torch.random.get_rng_state())

    def sample(self, logits):
        '''Sample a decoding of variables.'''
        while(True):
            value = []
            for i in range(logits.shape[0]):
                value.append(self.random.randint(0, logits.shape[1]-1))
            yield value

    def loss(self, logits):
        sampler = self.sample(logits)
        samples = [next(sampler) for _ in range(self.num_samples)]

        log_probs = torch.log_softmax(logits, dim=-1)
        satis_losses = tuple(map(lambda sample: decoding_loss(sample, log_probs),
                           filter(self.cond, samples)))
        satis_loss = torch.stack(satis_losses).logsumexp(dim=0)\
            if satis_losses else torch.tensor(0.0)

        viol_losses = tuple(map(lambda sample: decoding_loss(sample, log_probs),
                          filter(lambda v: not self.cond(v), samples)))
        viol_loss = torch.stack(viol_losses).logsumexp(dim=0)\
            if viol_losses else torch.tensor(0.0)

        log_prob_satis = satis_loss - torch.logsumexp(torch.stack((satis_loss, viol_loss)), dim=0)
        return -log_prob_satis


class WeightedSamplingSolver(SamplingSolver):
    def sample(self, logits):
        '''Sample a decoding of variables from a multinomial distribution parameterized by network probabilities'''

        probs = torch.softmax(logits, dim=-1)

        while(True):
            yield torch.multinomial(probs, num_samples=1, replacement=True).transpose_(0,1).flatten()
