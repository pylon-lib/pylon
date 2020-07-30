import torch
import itertools
from .solver import Solver

class SemanticSolver(Solver):

    def wmc(self, probs):
        models = filter(self.cond, itertools.product([0,1], repeat=probs.shape[-1]))

        wmc = 0
        for model in models:
            curr = 1
            for i, var in enumerate(model):
                curr *= probs[i] if var else 1 - probs[i]
            wmc += curr

        return wmc

    def loss(self, logits):
        probs = torch.softmax(logits, dim=-1)[:, -1]
        return -torch.log(self.wmc(probs))
