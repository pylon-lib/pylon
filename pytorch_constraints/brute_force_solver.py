import torch
import itertools

from .solver import Solver


class BruteForceSolver(Solver):
    '''Instantiates all possible decodings and checks whether the constraint is satisfied or not. Has two variations, depending on whether the loss is specified on satisfied decodings or unsatisfied decodings.'''

    def all_samples(self, probs):
        '''All possible decodings from the `probs` Tensor.'''
        yield from itertools.product(range(probs.shape[1]), repeat=probs.shape[0])

    def inst_loss(self, values, logits):
        '''Loss for a single decoded value of the variables.'''
        # sum of log prob
        # prob -> logprob = (0,1] -> (-inf, 0]
        #
        log_probs = torch.log_softmax(logits, dim=-1)
        loss = 0
        for log_prob, value in zip(log_probs, values):
            loss += log_prob[value]
        return loss

    def filter(self, value):
        '''Which variable values should we compute the loss over.'''
        raise NotImplementedError

    def reduce(self, losses):
        '''How do we aggregate the losses for each decoding.'''
        return sum(losses)

    def loss(self, logits):
        probs = torch.softmax(logits, dim=-1)
        samples = filter(self.filter, self.all_samples(probs))
        losses = map(lambda values: self.inst_loss(values, logits), samples)
        return self.reduce(losses)


class SatisfactionBruteForceSolver(BruteForceSolver):
    '''Add a loss that encourages the total probability over all possible decodings that satisfy the constraint.'''

    def filter(self, value):
        return self.cond(value)

    def reduce(self, losses):
        # we want to maximize this target
        return -torch.stack(tuple(losses)).logsumexp(dim=0)


class ViolationBruteForceSolver(BruteForceSolver):
    '''Add a loss that discourages the total probability over all possible decodings that violate the constraint.'''

    def filter(self, value):
        return not self.cond(value)

    def reduce(self, losses):
        return torch.stack(tuple(losses)).logsumexp(dim=0)
