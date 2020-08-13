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
            values = []
            for i in range(len(logits)):
                value = []
                for v in range(logits[i].shape[0]):
                    value.append(self.random.randint(0, logits[i].shape[1]-1))
                values.append(tuple(value))
            yield tuple(values)

    def loss(self, *logits):
        sampler = self.sample(logits)
        samples = [next(sampler) for _ in range(self.num_samples)]

        log_probs = [torch.log_softmax(logits[i], dim=-1) for i in range(len(logits))]
        satis_losses = tuple(map(lambda sample: decoding_loss(sample, log_probs),
                                 [s for s in samples if self.cond(*s)]))
        satis_loss = torch.stack(satis_losses).logsumexp(dim=0)\
            if satis_losses else torch.tensor(0.0)

        viol_losses = tuple(map(lambda sample: decoding_loss(sample, log_probs),
                                [s for s in samples if not self.cond(*s)]))
        viol_loss = torch.stack(viol_losses).logsumexp(dim=0)\
            if viol_losses else torch.tensor(0.0)

        log_prob_satis = satis_loss - torch.logsumexp(torch.stack((satis_loss, viol_loss)), dim=0)
        return -log_prob_satis


class WeightedSamplingSolver(SamplingSolver):
    def sample(self, logits):
        '''Sample a decoding of variables from a multinomial distribution parameterized by network probabilities'''

        probs = [torch.softmax(logits[i], dim=-1) for i in range(len(logits))]

        while(True):
            yield tuple([torch.multinomial(probs[i], num_samples=1, replacement=True).transpose_(0, 1).flatten() for i in range(len(logits))])
