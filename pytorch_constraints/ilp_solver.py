import torch
import pulp

from .solver import ASTLogicSolver
from .tree_node import TreeNodeVisitor


class ILPVisitor(TreeNodeVisitor):

    def visit_And(self, node, args):
        raise NotImplementedError


class ILPSolver(ASTLogicSolver):

    def inference(self, y_prob):
        m = pulp.LpProblem("ilp", pulp.LpMaximize)

        # variables
        y_var = []
        for i, prob in enumerate(y_prob):
            var = pulp.LpVariable(f'y{i}', lowBound=0, upBound=1, cat='Binary')
            # varnot = pulp.LpVariable(f'noty{i}', lowBound=0, upBound=1, cat='Binary')
            y_var.append([1 - var, var])

        # constraints
        m += self.bool_tree.prod_tnorm(y_var) == 1

        # objective
        obj = 0
        for prob, var in zip(y_prob, y_var):
            score = prob.log()
            obj += score[1] * var + score[0] * (1 - var)
        m += obj

        status = m.solve()

        sln = []
        for var in y_var:
            val = pulp.value(var)
            sln.append(val)

        return sln

    def sample(self, y_prob):
        return [self.inference(y_prob), ]

    def loss(self, logits):
        probs = torch.softmax(logits, dim=-1)
        infer = self.inference(probs)
        print(probs)
        print(infer)
        return -self.bool_tree.prod_tnorm(probs).log()
