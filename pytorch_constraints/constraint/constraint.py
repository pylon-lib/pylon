import torch
import itertools


class BaseConstraint:
    def cond(self, value):
        raise NotImplementedError

    def sample(self, probs):
        yield from itertools.product(range(probs.shape[1]), repeat=probs.shape[0])

    def loss(self, value, logit):
        raise NotImplementedError

    def reduce(self, losses):
        return sum(losses)

    def __call__(self, logits):
        probs = torch.softmax(logits, dim=-1)
        samples = self.sample(probs)
        losses = map(lambda values: self.loss(values, logits), samples)
        return self.reduce(losses)
