from .tree_node import *

import torch

from .solver import ASTLogicSolver
from .ast_visitor import *
from pysdd.sdd import SddManager, Vtree, WmcManager


class SemanticLossCircuitSolver(ASTLogicSolver):

    def loss(self, *logits):
        bool_tree = self.get_bool_tree()
        probs = [torch.softmax(logits[i], dim=-1) for i in range(len(logits))]
        vtree = Vtree(var_count=1)
        mgr = SddManager.from_vtree(vtree)
        sdd = SddVisitor().visit(bool_tree, mgr)
        return -self.prob(sdd, probs).log()

    # TODO: write a logprob solver that avoids underflow.
    def prob(self, sdd, lit_probs):
        if sdd.is_false():
            return 0.0
        elif sdd.is_true():
            return 1.0
        elif sdd.is_literal() and sdd.literal > 0:
            return lit_probs[0][sdd.literal-1][1]
        elif sdd.is_literal() and sdd.literal < 0:
            return lit_probs[0][-sdd.literal-1][0]
        elif sdd.is_decision():
            p = 0.0
            for prime, sub in sdd.elements():
                p += self.prob(prime, lit_probs) * self.prob(sub, lit_probs)
            return p
        else:
            raise ValueError("unknown type of SDD node")


class SddVisitor(TreeNodeVisitor):

    def visit_And(self, node, mgr):
        lv = self.visit(node.left, mgr)
        rv = self.visit(node.right, mgr)
        return lv & rv

    def visit_Or(self, node, mgr):
        lv = self.visit(node.left, mgr)
        rv = self.visit(node.right, mgr)
        return lv | rv

    def visit_Not(self, node, mgr):
        return ~self.visit(node.operand, mgr)

    def visit_IsEq(self, node, mgr):
        return self.visit(node.left, mgr).equiv(self.visit(node.right, mgr))

    def visit_Const(self, node, mgr):
        if node.value == True or node.value == 1:
            return mgr.true()
        elif node.value == False or node.value == 0:
            return mgr.false()
        else:
            raise NotImplementedError

    def visit_FunDef(self, node, mgr):
        return self.visit(node.return_node, mgr)

    def visit_VarList(self, node, mgr):
        if len(node.indices) > 1:
            raise NotImplementedError
        if node.arg.arg_pos != 0:
            raise NotImplementedError
        else:
            while mgr.var_count() < node.indices[0]+1:
                mgr.add_var_after_last()
            return mgr.literal(node.indices[0]+1)
