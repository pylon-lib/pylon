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
        samples = tuple(torch.distributions.categorical.Categorical(probs=probs[i]).sample((num_samples,))
                        for i in range(len(probs)))

        return list(zip(*samples))

    def loss(self, *logits, **kwargs):

        if kwargs.get('input') is not None:

            inputs = kwargs['input'].flatten() 
            num_classes = logits[0].shape[-1] 

            mul_mask = torch.ones(-1,num_classes) 
            mul_mask[inputs.nonzero(as_tuple=True)] = 0
            mul_mask = mul_mask.view(logits[0].shape)

            add_mask = torch.zeros(-1, num_classes) 
            add_mask[inputs.nonzero(as_tuple=True)] = -float('inf')
            tmp = add_mask[inputs.nonzero(as_tuple=True)]
            tmp.scatter_(-1, inputs[inputs.nonzero()] -1, 0)
            add_mask[inputs.nonzero(as_tuple=True)] = tmp
            add_mask = add_mask.view(logits[0].shape) 

            logits = tuple(logit * mul_mask + add_mask for logit in logits)

        llogits = len(logits) 
        log_probs = [torch.log_softmax(logits[i], dim=-1) for i in range(llogits)]

        samples = self.sample(logits, self.num_samples)
        losses = torch.stack([decoding_loss(sample, log_probs) for sample in samples])

        # TODO: remove extraneous torch.tensor wrapping. Currently in place for XOR example
        indices = torch.stack([torch.tensor(data=self.cond(*sample), dtype=torch.bool) for sample in samples])

        # decide whether to squeeze the last dim of losses
        #   the sizeof indices indicates whether the input is batched or not
        #   this can help to determin whether losses should be further aggregated
        if indices.shape != losses.shape:
            losses = losses.sum(-1)
        if indices.shape != losses.shape:
            raise Exception("Weird loss shape {0}, doesn't match label samples shape {1}.".format(losses.shape, indicates.shape))

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
