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
            return subsel.probs(probs)[..., [const.value]]

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

    def visit_ForallAlong(self, node, probs):
        tensor = self.visit(node.left, probs)
        dim = node.right.value if node.right is not None else -1
        return tensor.prod(dim)

    def visit_ExistsAlong(self, node, probs):
        # the formulation of disjunction for product t-norm doesn't easily generalize
        #    so use \not \forall_x (\not x)
        tensor = self.visit(node.left, probs)
        dim = node.right.value if node.right is not None else -1
        return 1 - (1 - tensor).prod(dim)


class ProductTNormLogicSolver(ASTLogicSolver):

    def loss(self, *logits, **kwargs):
        bool_tree = self.get_bool_tree()
        probs = [torch.softmax(logits[i], dim=-1) for i in range(len(logits))]
        rs = -ProductTNormVisitor().visit(bool_tree, probs).log()
        while len(rs.shape) > 2 and rs.shape[-1] == 1:
            rs = rs.squeeze(-1)
        return rs


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
        val = self.visit(node.expr, probs)
        return torch.relu(val.sum() - val.numel() + 1)

    def visit_ForallAlong(self, node, probs):
        tensor = self.visit(node.left, probs)
        dim = node.right.value if node.right is not None else -1
        # a generalized expression for conjunction
        #   instead of an aggregated version from pairwise operations
        return torch.relu(tensor.sum(dim) - tensor.shape[dim] + 1)

    def visit_ExistsAlong(self, node, probs):
        tensor = self.visit(node.left, probs)
        dim = node.right.value if node.right is not None else -1
        return 1 - torch.relu(1 - tensor.sum(dim))


class LukasiewiczTNormLogicSolver(ASTLogicSolver):

    def loss(self, *logits, **kwargs):
        bool_tree = self.get_bool_tree()
        probs = [torch.softmax(logits[i], dim=-1) for i in range(len(logits))]
        rs = -LukasiewiczTNormVisitor().visit(bool_tree, probs).log()
        while len(rs.shape) > 2 and rs.shape[-1] == 1:
            rs = rs.squeeze(-1)
        return rs


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

    def visit_ForallAlong(self, node, probs):
        tensor = self.visit(node.left, probs)
        dim = node.right.value if node.right is not None else -1
        return tensor.min(dim)[0]

    def visit_ExistsAlong(self, node, probs):
        tensor = self.visit(node.left, probs)
        dim = node.right.value if node.right is not None else -1
        return tensor.max(dim)[0]


class GodelTNormLogicSolver(ASTLogicSolver):

    def loss(self, *logits, **kwargs):
        bool_tree = self.get_bool_tree()
        probs = [torch.softmax(logits[i], dim=-1) for i in range(len(logits))]
        rs = -GodelTNormVisitor().visit(bool_tree, probs).log()
        while len(rs.shape) > 2 and rs.shape[-1] == 1:
            rs = rs.squeeze(-1)
        return rs
