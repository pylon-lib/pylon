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
        # samples = [self.sample(logits) for s in range(self.num_samples)]

        log_probs = torch.log_softmax(logits, dim=-1)
        satis_losses = map(lambda sample: decoding_loss(sample, log_probs),
                           filter(self.cond, samples))
        satis_losses = tuple(satis_losses)
        satis_loss = torch.stack(tuple(satis_losses)).logsumexp(dim=0)\
            if satis_losses else 0

        viol_losses = map(lambda sample: decoding_loss(sample, log_probs),
                          filter(lambda v: not self.cond(v), samples))
        viol_losses = tuple(viol_losses)
        viol_loss = torch.stack(tuple(viol_losses)).logsumexp(dim=0)\
            if viol_losses else 0

        # log_prob_satis = satis_loss - torch.logsumexp(satis_loss, viol_loss)
        # log_prob_viol = viol_loss - torch.logsumexp(satis_loss, viol_loss)
        # return -log_prob_satis
        return viol_loss - satis_loss


class WeightedSamplingSolver(SamplingSolver):
    def sample(self, logits):
        '''Sample a decoding of variables.'''
        # We're only interested in p, and not 1-p
        probs = torch.softmax(logits, dim=-1)[:, -1]
        while(True):
            yield torch.bernoulli(probs).long()
