import torch

from .constraint import BaseConstraint


class ViolationPenalty(BaseConstraint):
    def sample(self, probs):
        # sampling violating cases
        # e.g. XOR: 1,1 and 0,0
        yield from filter(lambda value: not self.cond(value), super().sample(probs))

    def loss(self, values, logits):
        # sum of log prob
        #
        log_probs = torch.log_softmax(logits, dim=-1)
        loss = 0
        for log_prob, value in zip(log_probs, values):
            loss += log_prob[value]
        return loss

    def reduce(self, losses):
        # we want to minimize this target (however this loss is unbounded)
        return torch.stack(tuple(losses)).logsumexp(dim=0)


class ViolationLambda(ViolationPenalty):
    def __init__(self, cond):
        super().__init__()
        self._cond = cond

    def cond(self, value):
        return self._cond(value)
