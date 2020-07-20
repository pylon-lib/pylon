import torch
import itertools

from .brute_force_solver import ViolationBruteForceSolver


class BaseConstraint:
    def __init__(self, cond, solver=None):
        self.cond = cond
        self.solver = solver
        self.solver.set_cond(cond)

    def loss(self, logits):
        return self.solver.loss(logits)

    def __call__(self, logits):
        return self.loss(logits)


def constraint(cond, solver=ViolationBruteForceSolver()):
    return BaseConstraint(cond, solver)
