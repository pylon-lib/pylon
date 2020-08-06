import torch

from .solver import ASTLogicSolver


class ProductTNormLogicSolver(ASTLogicSolver):

    def loss(self, logits):
        probs = torch.softmax(logits, dim=-1)
        return -self.bool_tree.prod_tnorm(probs).log()
