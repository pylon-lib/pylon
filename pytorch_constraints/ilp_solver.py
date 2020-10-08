import torch
import torch.nn.functional as F
import pulp

from .solver import ASTLogicSolver
from .tree_node import TreeNodeVisitor


class ILPVisitor(TreeNodeVisitor):
    def __init__(self):
        super().__init__()
        self._auto = -1

    @property
    def auto(self):
        self._auto += 1
        return self._auto

    def var(self, name):
        raise NotImplementedError

    def eq(self, expression):
        raise NotImplementedError

    def le(self, expression):
        raise NotImplementedError

    def visit_And(self, node, var):
        lv = self.visit(node.left, var)
        rv = self.visit(node.right, var)
        andv = self.var(f'and-{self.auto}')
        self.le(andv - lv)  # andv <= lv
        self.le(andv - rv)  # andv <= rv
        self.le(lv + rv - andv - 1)  # lv + rv <= andv + 1
        return andv

    def visit_Or(self, node, var):
        lv = self.visit(node.left, var)
        rv = self.visit(node.right, var)
        orv = self.var(f'or-{self.auto}')
        self.le(lv - orv)  # lv <= orv
        self.le(rv - orv)  # rv <= orv
        self.le(orv - lv - rv)  # orv <= lv + rv
        return orv

    def visit_Not(self, node, var):
        v = self.visit(node.operand, var)
        notv = self.var(f'not-{self.auto}')
        self.eq(v + notv - 1)  # v == 1 - notv
        return notv

    def visit_FunDef(self, node, var):
        return self.visit(node.return_node, var)

    def visit_VarList(self, node, var):
        return var[node.indices[0]]


class PulpILPVisitor(ILPVisitor):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def var(self, name):
        return pulp.LpVariable(f'visitor_{name}', lowBound=0, upBound=1, cat='Binary')

    def eq(self, expression):
        self.model += expression == 0

    def le(self, expression):
        self.model += expression <= 0


class ILPSolver(ASTLogicSolver):
    def inference(self, probs):
        bool_tree = self.get_bool_tree()
        m = pulp.LpProblem("ilp", pulp.LpMaximize)

        # variables
        y_var = []
        for i, prob in enumerate(probs):
            var = pulp.LpVariable(f'y{i}', lowBound=0, upBound=1, cat='Binary')
            # varnot = pulp.LpVariable(f'noty{i}', lowBound=0, upBound=1, cat='Binary')
            y_var.append(var)

        # constraints
        visitor = PulpILPVisitor(m)
        m += visitor.visit(bool_tree, y_var) == 1

        # objective
        obj = 0
        scores = probs.detach().log().cpu().numpy()
        for score, var in zip(scores, y_var):
            obj += score[1] * var + score[0] * (1 - var)
        m += obj

        status = m.solve()

        sln = []
        for var in y_var:
            val = pulp.value(var)
            sln.append(val)

        return sln

    def sample(self, probs):
        return [self.inference(probs), ]

    def loss(self, logits, targets=None):
        probs = torch.softmax(logits, dim=-1)
        samples = self.sample(probs)
        loss = 0
        if targets is not None:
            # supervised
            for sample in samples:
                weight = (targets != torch.tensor(sample)).float()
                targets = torch.stack([1-targets, targets], dim=-1).float()
                loss += F.binary_cross_entropy_with_logits(logits, targets, weight=weight)
        else:
            # unsupervised:
            for sample in samples:
                targets = torch.tensor(sample)
                targets = torch.stack([1-targets, targets], dim=-1).float()
                loss += F.binary_cross_entropy_with_logits(logits, targets)
        return loss
