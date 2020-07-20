import itertools


class Solver:

    def set_cond(self, cond):
        self.cond = cond

    def loss(self, logits):
        raise NotImplementedError
