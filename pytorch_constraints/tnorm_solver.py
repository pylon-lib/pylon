import torch
import numpy as np
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

    def visit_Implication(self, node, probs):
        lv = self.visit(node.left, probs)
        rv = self.visit(node.right, probs)
        # min(1, rv / lv) = 1 - relu(1 - rv / lv)
        return 1 - torch.relu(1 - rv / lv)

    def visit_SigmoidalImplication(self, node, probs):
        return sigmoidal_implication(self.visit_Implication, node, probs)

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

    def visit_VarUse(self, node, probs):
        # assume 0 is false
        return 1.0 - node.probs(probs)[0]

    def visit_IdentifierRef(self, node, probs):
        return self.visit(node.iddef.definition, probs)

    def visit_FunDef(self, node, probs):
        return self.visit(node.return_node, probs)


class ProductTNormLogicSolver(ASTLogicSolver):

    def loss(self, *logits, **kwargs):
        probs = [torch.softmax(logits[i], dim=-1) for i in range(len(logits))]
        return -ProductTNormVisitor().visit(self.bool_tree, probs).log()


class LukasiewiczTNormVisitor(TreeNodeVisitor):

    def visit_And(self, node, probs):
        lv = self.visit(node.left, probs)
        rv = self.visit(node.right, probs)
        return torch.min(lv, rv)

    def visit_Or(self, node, probs):
        lv = self.visit(node.left, probs)
        rv = self.visit(node.right, probs)

        # when input tensor has 0-dim, torch.min/max with constant doesn't work directly, so use relu
        # min(1,a+b) = 1-relu(0,1-a-b)
        return 1 - torch.relu(1 - lv - rv)

    def visit_Implication(self, node, probs):
        lv = self.visit(node.left, probs)
        rv = self.visit(node.right, probs)
        # min(1, 1 - lv + rv) = 1 - relu(lv - rv)
        return 1 - torch.relu(lv - rv)

    def visit_SigmoidalImplication(self, node, probs):
        return sigmoidal_implication(self.visit_Implication, node, probs)

    def visit_Not(self, node, probs):
        return 1.0 - self.visit(node.operand, probs)

    def visit_IsEq(self, node, probs):
        return ProductTNormVisitor().visit(node, probs)

    def visit_Constant(self, node, probs):
        return 1.0 if node.value else 0.0

    def visit_VarUse(self, node, probs):
        # assume 0 is false
        return 1.0 - node.probs(probs)[0]

    def visit_IdentifierRef(self, node, probs):
        return self.visit(node.iddef.definition, probs)

    def visit_FunDef(self, node, probs):
        return self.visit(node.return_node, probs)


class LukasiewiczTNormLogicSolver(ASTLogicSolver):

    def loss(self, *logits, **kwargs):
        probs = [torch.softmax(logits[i], dim=-1) for i in range(len(logits))]
        return -LukasiewiczTNormVisitor().visit(self.bool_tree, probs).log()


class GodelTNormVisitor(TreeNodeVisitor):

    def visit_And(self, node, probs):
        lv = self.visit(node.left, probs)
        rv = self.visit(node.right, probs)
        return torch.min(lv, rv)

    def visit_Or(self, node, probs):
        lv = self.visit(node.left, probs)
        rv = self.visit(node.right, probs)
        return torch.max(lv, rv)

    def visit_Implication(self, node, probs):
        lv = self.visit(node.left, probs)
        rv = self.visit(node.right, probs)
        # 1 if rv >= lv else rv
        rv_geq_lv = (rv >= lv).to(lv)
        return rv_geq_lv + (1 - rv_geq_lv) * rv

    def visit_SigmoidalImplication(self, node, probs):
        return sigmoidal_implication(self.visit_Implication, node, probs)

    def visit_Not(self, node, probs):
        return 1.0 - self.visit(node.operand, probs)

    def visit_IsEq(self, node, probs):
        return ProductTNormVisitor().visit(node, probs)

    def visit_Constant(self, node, probs):
        return 1.0 if node.value else 0.0

    def visit_VarUse(self, node, probs):
        # assume 0 is false
        return 1.0 - node.probs(probs)[0]

    def visit_IdentifierRef(self, node, probs):
        return self.visit(node.iddef.definition, probs)

    def visit_FunDef(self, node, probs):
        return self.visit(node.return_node, probs)


class GodelTNormLogicSolver(ASTLogicSolver):

    def loss(self, *logits, **kwargs):
        probs = [torch.softmax(logits[i], dim=-1) for i in range(len(logits))]
        return -GodelTNormVisitor().visit(self.bool_tree, probs).log()
