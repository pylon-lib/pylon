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
        value = []
        for i in range(logits.shape[0]):
            value.append(self.random.randint(0, logits.shape[1]-1))
        return value

    def loss(self, logits):
        log_probs = torch.log_softmax(logits, dim=-1)
        samples = [self.sample(logits) for s in range(self.num_samples)]
        print(samples)
        pos_losses = map(lambda sample: decoding_loss(sample, log_probs),
                         filter(self.cond, samples))
        pos_loss = torch.stack(tuple(pos_losses)).logsumexp(dim=0)
        neg_losses = map(lambda sample: decoding_loss(sample, log_probs),
                         filter(lambda v: not self.cond(v), samples))
        neg_loss = torch.stack(tuple(neg_losses)).logsumexp(dim=0)
        return neg_loss - pos_loss
