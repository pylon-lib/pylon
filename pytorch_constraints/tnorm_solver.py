import torch

from .solver import ASTLogicSolver


class ProductTNormLogicSolver(ASTLogicSolver):

    def loss(self, *logits):
        probs = [torch.softmax(logits[i], dim=-1) for i in range(len(logits))]
        return -self.bool_tree.prod_tnorm(probs).log()
