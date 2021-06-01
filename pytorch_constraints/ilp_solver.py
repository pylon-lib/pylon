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

    def visit_And(self, node, vars):
        lv = self.visit(node.left, vars)
        rv = self.visit(node.right, vars)
        andv = self.var(f'and-{self.auto}')
        self.le(andv - lv)  # andv <= lv
        self.le(andv - rv)  # andv <= rv
        self.le(lv + rv - andv - 1)  # lv + rv <= andv + 1
        return andv

    def visit_Or(self, node, vars):
        lv = self.visit(node.left, vars)
        rv = self.visit(node.right, vars)
        orv = self.var(f'or-{self.auto}')
        self.le(lv - orv)  # lv <= orv
        self.le(rv - orv)  # rv <= orv
        self.le(orv - lv - rv)  # orv <= lv + rv
        return orv

    def visit_Not(self, node, vars):
        v = self.visit(node.operand, vars)
        notv = self.var(f'not-{self.auto}')
        self.eq(v + notv - 1)  # v == 1 - notv
        return notv

    def visit_FunDef(self, node, vars):
        return self.visit(node.return_node, vars)

    def visit_VarList(self, node, vars):
        return vars[node.arg.arg_pos][node.indices[0]][1]

    def visit_IsEq(self, node, vars):
        lv = self.visit(node.left, vars)
        rv = self.visit(node.right, vars)
        iseqv = self.var(f'iseq-{self.auto}')
        self.le(lv + rv - iseqv - 1)  # TT -> T
        self.le(- lv - rv - iseqv + 1)  # FF -> T
        self.le(lv - rv + iseqv - 1)  # TF -> F
        self.le(- lv + rv + iseqv - 1)  # FT -> F
        return iseqv


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
    def score(self, prob):
        return prob.detach().log().cpu().numpy()

    def inference(self, *probs):
        bool_tree = self.get_bool_tree()
        m = pulp.LpProblem("ilp", pulp.LpMaximize)

        # variables
        vars = []
        for i, prob in enumerate(probs):
            var = []
            vars.append(var)
            for j, pr in enumerate(prob):
                var_values = []
                var.append(var_values)
                for k, p in enumerate(pr):
                    var_value = pulp.LpVariable(f'y_{i}_{j}_{k}', lowBound=0, upBound=1, cat='Binary')
                    var_values.append(var_value)
                m += sum(var_values) == 1

        # constraints
        visitor = PulpILPVisitor(m)
        m += visitor.visit(bool_tree, vars) == 1

        # objective
        obj = 0
        for prob, var in zip(probs, vars):
            score = self.score(prob)
            for pr, var_values, sc in zip(prob, var, score):
                for p, var_value, s in zip(pr, var_values, sc):
                    obj += s * var_value
        # for score, var in zip(scores, y_var):
        #     obj += score[1] * var + score[0] * (1 - var)
        m += obj

        status = m.solve()

        sln = []
        for var in vars:
            sln_var = []
            sln.append(sln_var)
            for var_values in var:
                sln_var_values = []
                sln_var.append(sln_var_values)
                for var_value in var_values:
                    val = pulp.value(var_value)
                    sln_var_values.append(val)

        return sln

    def sample(self, *probs):
        # batch mode: probs: argument(list), batch, varible, value
        # single mode: probs: argument(list), varible, value
        mode_shape_lens = set([len(prob.shape) for prob in probs])
        assert len(mode_shape_lens) == 1
        mode_shape_len = mode_shape_lens.pop()
        if mode_shape_len == 3:
            # batch mode:
            batch_probs = list(zip(*probs))
            return [self.inference(*probs_) for probs_ in batch_probs]
        elif mode_shape_len == 2:
            # single mode:
            return [self.inference(*probs), ]
        else:
            raise ValueError('Unknown {mode_shape_len}-dimensional tensor. 3-dim or 2-dim tensor expected.')

    def loss(self, *logits, targets=None):
        if not isinstance(targets, tuple) and targets is not None:
            targets = (targets,)
        probs = [torch.softmax(logit, dim=-1) for logit in logits]
        samples = self.sample(*probs)
        loss = 0
        if targets is not None:
            # supervised
            for sample in samples:
                for var, logit, target in zip(sample, logits, targets):
                    weight = (target != torch.tensor(var)).float()
                    target_binary = torch.zeros_like(logit)
                    target_binary.scatter_(1, target.unsqueeze(-1).long(), 1)
                    loss += F.binary_cross_entropy_with_logits(logit, target_binary, weight=weight)
        else:
            # unsupervised:
            for sample in samples:
                for var, logit in zip(sample, logits):
                    target_binary = torch.tensor(var)
                    loss += F.binary_cross_entropy_with_logits(logit, target_binary)
        return loss
