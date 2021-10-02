from pysdd.sdd import SddManager, Vtree
import torch

from .lazy_tensor import ConstShapedLazyTensor, LazyTensor
from .solver import Solver

class SemanticLossCircuitSolver(Solver):

    def __init__(self):
        self.sdd = None

    def loss(self, *logits):
        probs = [torch.softmax(logits[i], dim=-1) for i in range(len(logits))]
        if self.sdd is None:
            ys = [ConstShapedLazyTensor(i) for i in range(len(probs))]
            slt = self.cond(*ys)
            vtree = Vtree(var_count=1)
            mgr = SddManager.from_vtree(vtree)
            self.sdd = self.visit(slt, mgr)
        return -self.prob(self.sdd, probs).log()

    def visit(self, slt, mgr):
        """Visit a LazyTensor"""
        try:
            if isinstance(slt, ConstShapedLazyTensor):
                return slt
            lv = self.visit(slt.args[0], mgr)
            if len(slt.args) == 1:
                rv = None
            elif not isinstance(slt.args[1], LazyTensor):
                assert slt.function == torch.Tensor.__getitem__
                rv = slt.args[1]
            else:
                rv = self.visit(slt.args[1], mgr)
            return self.getcircuit(lv, rv, slt.function, mgr)

        except KeyError:
            raise Exception(f'{slt.function.__name__} not in SWITCHER')

    def getcircuit(self, lv, rv, function, mgr):
        return {
            torch.logical_not: lambda: ~lv,
            torch.logical_and: lambda: lv & rv,
            torch.logical_or: lambda: lv | rv,
            torch.eq: lambda: lv.equiv(rv),
            torch.ne: lambda: ~lv.equiv(rv),
            torch.Tensor.__getitem__: lambda: self.getitem(lv, rv, mgr),
        }[function]()

    def getitem(self, lv, rv, mgr):
        assert isinstance(rv, int)
        # assumes lv is first variable
        while mgr.var_count() < rv + 1:
            mgr.add_var_after_last()
        return mgr.literal(rv + 1)

    # TODO: write a logprob solver that avoids underflow.
    def prob(self, sdd, lit_probs):
        assert len(lit_probs)==1
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
