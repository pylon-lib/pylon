from abc import abstractmethod
from enum import Enum
import torch

from .lazy_tensor import AbstractLazyTensor, LazyTensor, ConstShapedLazyTensor
from .solver import Solver


class Type(Enum):
    CONST = 1
    SUB_SELECT = 2
    COND = 3


def get_type(v):
    """
    returns the type of node v, helping TNormSolver.eq_tnorm determine the
    correct eq computation
    """
    if not issubclass(v.__class__, AbstractLazyTensor) or isinstance(v, ConstShapedLazyTensor):
        return Type.CONST
    assert isinstance(v, LazyTensor)
    if v.function == torch.Tensor.__getitem__:
        if issubclass(v.args[1].__class__, AbstractLazyTensor):
            # the key is a lazy tensor that represents x==1 e.g. y[x==1]
            return Type.COND
        # the key is an integer  e.g. y[1]
        return Type.SUB_SELECT

class TNormSolver(Solver):
    def base_tnorm(self, args, function, probs):
        tnorm_dict = {}
        # args are arranged as (lv, rv[0], rv[1], ...)
        #   if lv is None, this is probably a non-member call e.g. torch.cat(...) where torch is the lazy torch object
        lv = self.visit(args[0])(probs) if args[0] is not None else None
        rv = None

        # eq and indexing needs special handling
        if lv is not None and len(args) == 2 and function in [torch.eq, torch.Tensor.__getitem__]:
            rv = self.visit(args[1])(probs) if isinstance(args[1], LazyTensor) else args[1]
            # torch.Tensor.__getitem__
            if function == torch.Tensor.__getitem__:
                tnorm_dict[torch.Tensor.__getitem__] = lambda: lv[rv]
            elif function == torch.eq:
                # need to know the type of left and right nodes
                l_type = get_type(args[0])
                r_type = get_type(args[1])

                # map torch.eq to the corresponding formula
                if l_type == Type.CONST and r_type == Type.CONST:
                    tnorm_dict[torch.eq] = 1.0 if lv == rv else 0.0

                elif l_type == Type.SUB_SELECT and r_type == \
                        Type.SUB_SELECT:
                    tnorm_dict[torch.eq] = lambda: (lv * rv).sum(dim=-1)

                elif (l_type == Type.CONST and r_type == Type.SUB_SELECT) \
                        or (
                        l_type == Type.SUB_SELECT and r_type == Type.CONST):
                    subsel = rv if r_type == Type.SUB_SELECT else lv
                    const = rv if r_type == Type.CONST else lv
                    tnorm_dict[torch.eq] = lambda: subsel[const]

                elif (r_type == Type.COND and l_type == Type.CONST) \
                        or (l_type == Type.COND and r_type == Type.CONST):

                    cond = args[1] if r_type == Type.COND else args[0]
                    const = args[1] if r_type == Type.CONST else args[0]

                    assert len(cond.args) == 2
                    cond_expr = cond.args[1]
                    # create a lazy tensor for the arg of the cond == const
                    cond_arg_equal_const = cond.args[0] == const

                    tnorm_dict[torch.eq] = self.visit(
                        cond_arg_equal_const.logical_or(
                            cond_expr.logical_not())
                        )(probs)

                else:
                    raise NotImplementedError

        elif len(args) >= 2:
            rv = [self.visit(p)(probs) if isinstance(p, LazyTensor) else p for p in args[1:]]
            # push rv to lv's device
            if isinstance(lv, torch.Tensor):
                rv = [p.to(lv.device) if isinstance(p, torch.Tensor) else p for p in rv]
            rv = rv[0] if len(rv) == 1 else rv

        return {
            **tnorm_dict,
            #
            torch.clone: lambda: lv.clone(),
            torch.ne: lambda: (1 - lv * rv).sum(dim=-1),    # TODO, this is wrong
            # Tensor functions
            torch.Tensor.logical_not: lambda: 1.0 - lv,
            torch.Tensor.size: lambda: lv.size(),
            torch.Tensor.squeeze: lambda: lv.squeeze(rv),
            torch.Tensor.unsqueeze: lambda: lv.unsqueeze(rv),
            torch.Tensor.expand_as: lambda: lv.expand_as(rv),
            torch.Tensor.diagonal: lambda: lv.diagonal(rv[0], rv[1], rv[2]),
            torch.Tensor.zero_: lambda: lv.zero_(),
            torch.Tensor.add_: lambda: lv.add_(rv),
            torch.Tensor.sub_: lambda: lv.sub_(rv),
            torch.Tensor.mul_: lambda: lv.mul_(rv),
            torch.Tensor.div_: lambda: lv.div_(rv),
            torch.Tensor.add: lambda: lv.add(rv),
            torch.Tensor.sub: lambda: lv.sub(rv),
            torch.Tensor.mul: lambda: lv.mul(rv),
            torch.Tensor.div: lambda: lv.div(rv),
            torch.Tensor.floor_divide: lambda: lv.floor_divide(rv),
            torch.Tensor.remainder: lambda: lv.remainder(rv),
            torch.Tensor.bmm: lambda: lv.bmm(rv),
            torch.Tensor.mm: lambda: lv.mm(rv),
            torch.Tensor.sqrt: lambda: lv.sqrt(),
            torch.Tensor.rsqrt: lambda: lv.rsqrt(),
            torch.Tensor.log: lambda: lv.log(),
            torch.Tensor.exp: lambda: lv.exp(),
            torch.Tensor.tile: lambda: lv.tile(*rv),
            torch.Tensor.sigmoid: lambda: lv.sigmoid(),
            torch.Tensor.softmax: lambda: lv.softmax(rv),
            torch.Tensor.relu: lambda: lv.relu(),
            torch.Tensor.min: lambda: lv.min(rv) if rv is not None else lv.min(),
            torch.Tensor.max: lambda: lv.max(rv) if rv is not None else lv.max(),
            torch.Tensor.sum: lambda: lv.sum(rv) if rv is not None else lv.sum(),
            torch.Tensor.logsumexp: lambda: lv.logsumexp(rv),
            torch.Tensor.float: lambda: lv.float(),
            torch.Tensor.half: lambda: lv.half(),
            torch.Tensor.double: lambda: lv.double(),
            torch.Tensor.int: lambda: lv.int(),
            torch.Tensor.long: lambda: lv.long(),
            torch.Tensor.short: lambda: lv.short(),
            torch.Tensor.byte: lambda: lv.byte(),
            torch.Tensor.bool: lambda: lv.bool(),
            torch.Tensor.masked_select: lambda: lv.masked_select(rv),
            # torch.xxx calls where lv is None
            torch.zeros: lambda: torch.zeros(*rv),
            torch.ones: lambda: torch.ones(*rv),
            torch.eye: lambda: torch.eye(rv),
            torch.cat: lambda: torch.cat(rv[0], dim=rv[1]),  # rv: (tensors, dim)
            torch.stack: lambda: torch.stack(rv[0], dim=rv[1]),  # rv: (tensors, dim)
            torch.tile: lambda: torch.tile(rv[0], rv[1]),
            torch.squeeze: lambda: rv[0].squeeze(rv[1]),
            torch.unsqueeze: lambda: rv[0].unsqueeze(rv[1]),
            torch.randn: lambda: torch.randn(*rv),
            torch.sum: lambda: rv[0].sum(rv[1]) if rv[1] is not None else rv[0].sum(),
            torch.min: lambda: rv[0].min(rv[1])[0] if rv[1] is not None else rv[0].min(),
            torch.max: lambda: rv[0].max(rv[1])[0] if rv[1] is not None else rv[0].max(),
            torch.add: lambda: rv[0].add(rv[1]),
            torch.sub: lambda: rv[0].sub(rv[1]),
            torch.mul: lambda: rv[0].mul(rv[1]),
            torch.div: lambda: rv[0].div(rv[1]),
            torch.floor_divide: lambda: rv[0].floor_divide(rv[1]),
            torch.sqrt: lambda: rv[0].sqrt(),
            torch.rsqrt: lambda: rv[0].rsqrt(),
            torch.mm: lambda: rv[0].mm(rv[1]),
            torch.bmm: lambda: rv[0].bmm(rv[1]),
            torch.log: lambda: rv[0].log(),
            torch.exp: lambda: rv[0].exp(),
            torch.relu: lambda: torch.relu(rv[0]),
            torch.sigmoid: lambda: rv[0].sigmoid(),
            torch.softmax: lambda: rv[0].softmax(rv[1]),
            torch.logsumexp: lambda: rv[0].logsumexp(rv[1]),
            torch.masked_select: lambda: rv[0].masked_select(rv[1]),
        }, lv, rv

    @abstractmethod
    def gettnorm(self, args, function, probs):
        print("Some implementation")

    def __init__(self):
        self.cond = None
        self.tensor_comp = None

    def visit(self, slt):
        """Visit a LazyTensor"""

        # probs: tensor
        def helper(probs):
            if not isinstance(slt, AbstractLazyTensor):
                return slt
            if isinstance(slt, ConstShapedLazyTensor):
                return probs[slt.index]     # lambda probs: probs[slt.index]
            try:
                return self.gettnorm(slt.args, slt.function, probs)

            except KeyError:
                raise Exception(f'{slt.function.__name__} not in TNORM_SWITCHER')

        return helper

    def loss(self, *logits, **kwargs):
        probs = [torch.softmax(logits[i], dim=-1) for i in range(len(logits))]

        if self.tensor_comp is None:
            ys = [ConstShapedLazyTensor(i) for i in range(len(probs))]
            slt = self.cond(*ys)
            self.tensor_comp = self.visit(slt)
        tensor = self.tensor_comp(probs)
        return -tensor.log()


class ProductTNormSolver(TNormSolver):
    def gettnorm(self, args, function, probs):
        def AND(t, dim):
            return t.prod(dim) if dim is not None else t.prod()
        def OR(t, dim):
            return (1-(1-t).prod(dim)) if dim is not None else (1-(1-t).prod())
        tnorm_dict, lv, rv  = self.base_tnorm(args, function, probs)
        return {
            **tnorm_dict,
            torch.Tensor.logical_not: lambda: 1.0 - lv,
            torch.Tensor.logical_and: lambda: lv * rv,
            torch.Tensor.logical_or: lambda: lv + rv - lv * rv,
            torch.eq: lambda: (lv * rv).sum(dim=-1),
            torch.ne: lambda: (1 - lv * rv).sum(dim=-1),
            torch.Tensor.__getitem__: lambda: lv[rv],
            torch.le: lambda: 1 - torch.relu(1 - rv / lv),  # lv implies rv
            torch.ge: lambda: 1 - torch.relu(1 - lv / rv),  # rv implies lv
            torch.Tensor.all: lambda: AND(lv, rv),  # lv is the tensor, rv is the dim
            torch.Tensor.any: lambda: OR(lv, rv),
            torch.all: lambda: AND(rv[0], rv[1]),
            torch.any: lambda: OR(rv[0], rv[1])
        }[function]()

class LukasiewiczTNormSolver(TNormSolver):
    def gettnorm(self, args, function, probs):
        def AND(t, dim):
            return (torch.relu(t.sum(dim) - t.shape[dim] + 1)) if dim is not None else (torch.relu(t.sum() - t.numel() + 1))
        def OR(t, dim):
            return (1 - torch.relu(1 - t.sum(dim))) if dim is not None else (1 - torch.relu(1 - t.sum()))
        tnorm_dict, lv, rv = self.base_tnorm(args, function, probs)
        return {
            **tnorm_dict,
            torch.Tensor.logical_and: lambda: torch.relu(lv + rv - 1),
            torch.Tensor.logical_or: lambda: 1 - torch.relu(1 - lv - rv),
            torch.logical_and: lambda: torch.relu(rv[0] + rv[1] - 1),
            torch.logical_or: lambda: 1 - torch.relu(1 - rv[0] - rv[1]),
            torch.eq: lambda: (lv * rv).sum(dim=-1),
            torch.ne: lambda: (1 - lv * rv).sum(dim=-1),
            torch.Tensor.__getitem__: lambda: lv[rv],
            torch.le: lambda: (rv >= lv).to(lv) + (1 - (rv >= lv).to(lv)) * rv,
            torch.ge: lambda: (lv >= rv).to(rv) + (1 - (lv >= rv).to(rv)) * lv,
            torch.Tensor.all: lambda: AND(lv, rv),
            torch.Tensor.any: lambda: OR(lv, rv),
            torch.all: lambda: AND(rv[0], rv[1]),
            torch.any: lambda: OR(rv[0], rv[1])
        }[function]()


class GodelTNormSolver(TNormSolver):
    def gettnorm(self, args, function, probs):
        def AND(t, dim):
            return t.min(dim)[0] if dim is not None else t.min()
        def OR(t, dim):
            return t.max(dim)[0] if dim is not None else t.max()
        tnorm_dict, lv, rv = self.base_tnorm(args, function, probs)
        return {
            **tnorm_dict,
            torch.Tensor.logical_and: lambda: torch.min(lv,rv),
            torch.Tensor.logical_or: lambda: torch.max(lv, rv),
            torch.logical_and: lambda: torch.min(rv[0],rv[1]),
            torch.logical_or: lambda: torch.max(rv[0],rv[1]),
            torch.eq: lambda: (lv * rv).sum(dim=-1),
            torch.ne: lambda: (1 - lv * rv).sum(dim=-1),
            torch.Tensor.__getitem__: lambda: lv[rv],
            torch.le: lambda: 1 - torch.relu(lv - rv),
            torch.ge: lambda: 1 - torch.relu(rv - lv),
            torch.Tensor.all: lambda: AND(lv, rv),  # lv is the tensor, rv is the dim
            torch.Tensor.any: lambda: OR(lv, rv),
            torch.all: lambda: AND(rv[0], rv[1]),
            torch.any: lambda: OR(rv[0], rv[1])
        }[function]()


#class LukasiewiczTNormSolver(TNormSolver):
#    def gettnorm(self, lv, l_type, rv, r_type, function):
#        return {
#            torch.logical_and: lambda: torch.relu(lv + rv - 1),
#            torch.logical_or: lambda: 1 - torch.relu(1 - lv - rv),
#            torch.ne: lambda: (1 - lv * rv).sum(dim=-1),
#            torch.Tensor.__getitem__: lambda: lv[rv],
#        }[function]()
#
#
#class GodelTNormSolver(TNormSolver):
#    def gettnorm(self, lv, l_type, rv, r_type, function):
#        return {
#            torch.logical_and: lambda: torch.min(lv,rv),
#            torch.logical_or: lambda: torch.max(lv, rv),
#            torch.ne: lambda: (1 - lv * rv).sum(dim=-1),
#            torch.Tensor.__getitem__: lambda: lv[rv],
#        }[function]()

