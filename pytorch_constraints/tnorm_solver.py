import torch

from .solver import ASTLogicSolver
from .ast_visitor import And, Or, VariableUse, Not


class TNormLogicSolver(ASTLogicSolver):

    def _visit_nodes(self, node, probs):
        if type(node) == VariableUse:
            return probs[node.index][1]
        if type(node) == Not:
            return 1.0 - self._visit_nodes(node.operand, probs)
        # assume And or Or
        lv = self._visit_nodes(node.left, probs)
        rv = self._visit_nodes(node.right, probs)
        if type(node) == And:
            return lv * rv
        if type(node) == Or:
            return lv + rv - lv * rv
        raise NotImplementedError

    def loss(self, logits):
        probs = torch.softmax(logits, dim=-1)
        return -self._visit_nodes(self.bool_tree, probs).log()
