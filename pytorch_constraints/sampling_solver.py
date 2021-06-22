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

    def sample(self, logits, num_samples):
        '''Sample a decoding of variables.'''

        probs = [torch.tensor(1/logits[i].shape[-1], device=logits[i].device).expand_as(logits[i])
                for i in range(len(logits))]

        # samples Shape: len(logits) x num_samples x logits[i].shape[:-1]
        samples = tuple(torch.distributions.categorical.Categorical(probs=probs[i]).sample((num_samples,))
                        for i in range(len(probs)))

        return list(zip(*samples))

    def loss(self, *logits, **kwargs):

        if kwargs.get('input') is not None:
            logits = list(logits)
            for i, inputs in enumerate(kwargs['input']):
                inputs = inputs.flatten()

                mul_mask = torch.ones(9, 9).to(logits[i].device)
                mul_mask[inputs.nonzero(as_tuple=True)] = 0

                add_mask = torch.zeros(9, 9).to(logits[i].device) 
                add_mask[inputs.nonzero(as_tuple=True)] = -float('inf')
                add_mask[inputs.nonzero(as_tuple=False), inputs[inputs.nonzero(as_tuple=False)] - 1] = 0

                logits[i] = logits[i] * mul_mask + add_mask

        llogits = len(logits) 
        log_probs = [torch.log_softmax(logits[i], dim=-1) for i in range(llogits)]
            
        # list of length num_samples. Each element
        # is a tuple of lenght klogits (i.e. the
        # number of classifiers) and each element of the
        # tuple is of length batch_size
        samples = self.sample(logits, self.num_samples)

        indices = torch.stack([self.cond(*sample, kwargs) for sample in samples])

        losses = torch.stack([decoding_loss(sample, log_probs) for sample in samples])

        sat_losses = losses.clone()
        sat_losses[~indices] = -float('inf')

        loss = sat_losses.logsumexp(dim=0) - losses.logsumexp(dim=0)
        return -loss.sum()


class WeightedSamplingSolver(SamplingSolver):

    def sample(self, logits, num_samples):
        '''Sample a decoding of variables from a multinomial distribution parameterized by network probabilities'''

        # samples Shape: len(logits) x num_samples x logits[i].shape[:-1]
        samples = tuple(torch.distributions.categorical.Categorical(logits=logits[i]).sample((num_samples,))
                        for i in range(len(logits)))

        return list(zip(*samples))
