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

    def sample(self, logits, num_samples):
        '''Sample a decoding of variables.'''

        probs = [torch.tensor(1/logits[i].shape[-1]).expand_as(logits[i]) for i in range(len(logits))]

        # samples Shape: len(logits) x num_samples x logits[i].shape[:-1]
        samples = tuple(torch.distributions.categorical.Categorical(probs=probs[i]).sample((num_samples,))\
                for i in range(len(probs)))

        return list(zip(*samples))

    def loss(self, *logits, **kwargs):

        llogits = len(logits) 
        log_probs = [torch.log_softmax(logits[i], dim=-1) for i in range(llogits)]

        samples = self.sample(logits, self.num_samples)
        losses = torch.stack([decoding_loss(sample, log_probs) for sample in samples])

        # TODO: remove extraneous torch.tensor wrapping. Currently in place for XOR example
        indices = torch.stack([torch.tensor(data=self.cond(*sample), dtype=torch.bool) for sample in samples])


        #loss = torch.tensor([losses[:, i][indices[:, i]].logsumexp(dim=0) for i in range(log_probs[0].shape[0])]) - losses.logsumexp(dim=0)
        #return -loss.sum()

        loss = losses[indices].logsumexp(dim=0) - losses.logsumexp(dim=tuple(i for i in range(len(losses.shape))))
        return -loss

class WeightedSamplingSolver(SamplingSolver):

    def sample(self, logits, num_samples):
        '''Sample a decoding of variables from a multinomial distribution parameterized by network probabilities'''

        # samples Shape: len(logits) x num_samples x logits[i].shape[:-1]
        samples = tuple(torch.distributions.categorical.Categorical(logits=logits[i]).sample((num_samples,))\
                for i in range(len(logits)))

        return list(zip(*samples))
