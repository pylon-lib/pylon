import torch

from .solver import ASTLogicSolver
from .tree_node import *


class ProductTNormVisitor(TreeNodeVisitor):

    def visit_And(self, node, probs):
        lv = self.visit(node.left, probs)
        rv = self.visit(node.right, probs)
        return lv * rv

    def visit_Or(self, node, probs):
        lv = self.visit(node.left, probs)
        rv = self.visit(node.right, probs)
        return lv + rv - lv * rv

    def visit_Not(self, node, probs):
        return 1.0 - self.visit(node.operand, probs)

    def visit_IsEq(self, node, probs):
        if isinstance(node.left, VarUse) and isinstance(node.right, Const):
            return node.left.probs(probs)[node.right.value]
        elif isinstance(node.left, Const) and isinstance(node.right, VarUse):
            return node.right.probs(probs)[node.left.value]
        elif isinstance(node.left, Const) and isinstance(node.right, Const):
            return 1.0 if node.left.value == node.right.value else 0.0
        elif isinstance(node.left, VarUse) and isinstance(node.right, VarUse):
            return (node.left.probs(probs)*node.right.probs(probs)).sum()
        else:
            raise NotImplementedError

    def visit_Constant(self, node, probs):
        return 1.0 if node.value else 0.0

    def visit_IdentifierRef(self, node, probs):
        return self.visit(node.iddef.definition, probs)

    def visit_FunDef(self, node, probs):
        return self.visit(node.return_node, probs)


class ProductTNormLogicSolver(ASTLogicSolver):

    def loss(self, *logits):
        probs = [torch.softmax(logits[i], dim=-1) for i in range(len(logits))]
        return -ProductTNormVisitor().visit(self.bool_tree, probs).log()


class LukasiewiczTNormLogicSolver(ASTLogicSolver):

    def loss(self, *logits):
        probs = [torch.softmax(logits[i], dim=-1) for i in range(len(logits))]
        return -self.bool_tree.lukasiewicz_tnorm(probs).log()


class GodelTNormLogicSolver(ASTLogicSolver):

    def loss(self, *logits):
        probs = [torch.softmax(logits[i], dim=-1) for i in range(len(logits))]
        return -self.bool_tree.godel_tnorm(probs).log()
