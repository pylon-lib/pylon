import torch

from .solver import ASTLogicSolver

class SemanticLossSddSolver(ASTLogicSolver):

    def loss(self, *logits):
        raise NotImplementedError

