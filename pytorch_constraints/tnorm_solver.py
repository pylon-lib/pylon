import torch

from .solver import ASTLogicSolver


class TNormLogicSolver(ASTLogicSolver):

    def loss(self, logits):
        probs = torch.softmax(logits, dim=-1)
        return -self.bool_tree.tnorm(probs).log()
