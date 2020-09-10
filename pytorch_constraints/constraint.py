import torch
import itertools

from .brute_force_solver import SatisfactionBruteForceSolver


class BaseConstraint:
    def __init__(self, cond, solver=None):
        self.cond = cond
        self.solver = solver
        self.solver.set_cond(cond)

    def loss(self, *logits, **kwargs):
        l = self.solver.loss(*logits, **kwargs)
        if len(l.shape) == 1:
            l = l[0]
        if len(l.shape) == 2:
            l = l[0, 0]
        return l

    def __call__(self, *logits, **kwargs):
        '''Return the differentiable loss for the constraint given the logits of the variables.'''
        return self.loss(*logits, **kwargs)


def constraint(cond, solver=SatisfactionBruteForceSolver()):
    '''Create a constraint for a given boolean condition.'''
    return BaseConstraint(cond, solver)
