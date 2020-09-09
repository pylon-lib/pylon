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

        weights = [torch.ones_like(logits[i]) for i in range(len(logits))]

        samples = tuple([torch.multinomial(weights[i], num_samples=num_samples, replacement=True).transpose_(0, 1) for i in range(len(logits))])
        return list(zip(*samples))

    def loss(self, *logits):

        shapes = [logits[i].shape for i in range(len(logits))]
        for i in range(len(logits)):
            #TODO: add logits= to the following line
            logits[i].reshape(-1, shapes[i][-1])

        samples = self.sample(logits, self.num_samples)

        log_probs = [torch.log_softmax(logits[i], dim=-1) for i in range(len(logits))]

        indices = [torch.tensor(data=self.cond(*sample), dtype=bool) for sample in samples]

        llogits = len(logits) 

        satis_losses = [decoding_loss(tuple(sample[i][indices[j]] for i in range(llogits)),\
                tuple(log_probs[i][indices[j]] for i in range(llogits))) for j, sample in enumerate(samples)\
                if indices[j].any()]

        viol_losses = [decoding_loss(tuple(sample[i][~indices[j]] for i in range(llogits)),\
                tuple(log_probs[i][~indices[j]] for i in range(llogits))) for j, sample in enumerate(samples)\
                if not indices[j].all()]

        satis_loss = torch.stack(satis_losses).logsumexp(dim=0)\
            if satis_losses else torch.tensor(0.0)

        viol_loss = torch.stack(viol_losses).logsumexp(dim=0)\
            if viol_losses else torch.tensor(0.0)

        log_prob_satis = satis_loss - torch.logsumexp(torch.stack((satis_loss, viol_loss)), dim=0)
        return -log_prob_satis


class WeightedSamplingSolver(SamplingSolver):

    def sample(self, logits, num_samples):
        '''Sample a decoding of variables from a multinomial distribution parameterized by network probabilities'''

        shapes = [logits[i].shape for i in range(len(logits))]
        logits = [logits[i].reshape(-1, shapes[i][-1]) for i in range(len(logits))]
        probs = [torch.softmax(logits[i], dim=-1) for i in range(len(logits))]

        samples = tuple([torch.multinomial(probs[i], num_samples=num_samples, replacement=True).transpose_(0, 1) for i in range(len(logits))])
        samples = [sample.reshape(num_samples, *shapes[i][:-1]) for i, sample in enumerate(samples)]

        return list(zip(*samples))
