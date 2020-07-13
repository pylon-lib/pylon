import torch

from .constraint import BaseConstraint


class SatisfactionPenalty(BaseConstraint):
    def sample(self, probs):
        # sampling satisfacting cases
        # e.g. XOR: 1,0 and 0,1
        yield from filter(self.cond, super().sample(probs))

    def loss(self, values, logits):
        # sum of log prob
        # prob -> logprob = (0,1] -> (-inf, 0]
        #
        log_probs = torch.log_softmax(logits, dim=-1)
        loss = 0
        for log_prob, value in zip(log_probs, values):
            loss += log_prob[value]
        return loss

    def reduce(self, losses):
        # we want to maximize this target
        return -torch.stack(tuple(losses)).logsumexp(dim=0)


class SatisfactionLambda(SatisfactionPenalty):
    def __init__(self, cond):
        super().__init__()
        self._cond = cond

    def cond(self, value):
        return self._cond(value)
