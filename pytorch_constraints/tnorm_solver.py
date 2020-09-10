import torch
import numpy as np
from .solver import ASTLogicSolver
from .tree_node import *


class TNormTreeNodeVisitor(TreeNodeVisitor):

    def visit_Not(self, node, probs):
        return 1.0 - self.visit(node.operand, probs)

    def visit_Constant(self, node, probs):
        return 1.0 if node.value else 0.0

    def visit_VarList(self, node, probs):
        # assume 0 is false
        return 1.0 - node.probs(probs)[:, 0]

    def visit_List(self, node, probs):
        return torch.stack([self.visit(elt, probs) for elt in node.elts])

    def visit_Exists(self, node, probs):
        return self.visit(Not(Forall(Not(node.expr))), probs)

    def visit_Arg(self, node, probs):
        return 1.0 - node.probs(probs)[:, 0]

    def visit_IsEq(self, node, probs):

        def is_constant(n): return isinstance(n, Const)
        def is_subselect(n): return isinstance(n, Arg) or isinstance(n, VarList)
        def is_cond(n): return isinstance(n, VarCond)

        (l_const, r_const) = (is_constant(node.left), is_constant(node.right))
        if l_const and r_const:
            return 1.0 if node.left.value == node.right.value else 0.0

        (l_subsel, r_subsel) = (is_subselect(node.left), is_subselect(node.right))
        if l_subsel and r_subsel:
            return (node.left.probs(probs)*node.right.probs(probs)).sum()

        if (l_const and r_subsel) or (r_const and l_subsel):
            subsel = node.right if r_subsel else node.left
            const = node.right if r_const else node.left
            return subsel.probs(probs)[:, [const.value]]

        (l_cond, r_cond) = (is_cond(node.left), is_cond(node.right))
        if (r_cond and l_const) or (r_const and l_cond):
            cond = node.right if r_cond else node.left
            const = node.right if r_const else node.left
            return self.visit(Or(Not(cond.expr), IsEq(cond.arg, const)), probs)

        raise NotImplementedError

    def visit_IdentifierRef(self, node, probs):
        return self.visit(node.iddef.definition, probs)

    def visit_FunDef(self, node, probs):
        return self.visit(node.return_node, probs)


class ProductTNormVisitor(TNormTreeNodeVisitor):

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

    def visit_Forall(self, node, probs):
        return self.visit(node.expr, probs).prod()


class ProductTNormLogicSolver(ASTLogicSolver):

    def loss(self, *logits, **kwargs):
        probs = [torch.softmax(logits[i], dim=-1) for i in range(len(logits))]
        tree_prob = ProductTNormVisitor().visit(self.bool_tree, probs)
        return -tree_prob.log()


class LukasiewiczTNormVisitor(TNormTreeNodeVisitor):

    def visit_And(self, node, probs):
        lv = self.visit(node.left, probs)
        rv = self.visit(node.right, probs)
        return torch.relu(lv + rv - 1)

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

    def visit_Forall(self, node, probs):
        return self.visit(node.expr, probs).min()


class LukasiewiczTNormLogicSolver(ASTLogicSolver):

    def loss(self, *logits, **kwargs):
        probs = [torch.softmax(logits[i], dim=-1) for i in range(len(logits))]
        return -LukasiewiczTNormVisitor().visit(self.bool_tree, probs).log()


class GodelTNormVisitor(TNormTreeNodeVisitor):

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

    def visit_Forall(self, node, probs):
        return self.visit(node.expr, probs).min()


class GodelTNormLogicSolver(ASTLogicSolver):

    def loss(self, *logits, **kwargs):
        probs = [torch.softmax(logits[i], dim=-1) for i in range(len(logits))]
        return -GodelTNormVisitor().visit(self.bool_tree, probs).log()
