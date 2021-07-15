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
        lv = self.visit(args[0])(probs)
        rv = None

        if len(args) == 2:
            # torch.Tensor.__getitem__
            if function == torch.Tensor.__getitem__:
                tnorm_dict[torch.Tensor.__getitem__] = lambda: lv[rv]
            else:
                # rv could be a lazy tensor
                rv = self.visit(args[1])(probs)

                #  torch.eq
                if function == torch.eq:
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

        return {
            **tnorm_dict,
            # other default functions
            torch.logical_not: lambda: 1.0 - lv,
            torch.logical_and: lambda: lv * rv,
            torch.ne: lambda: (1 - lv * rv).sum(dim=-1),
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
        tnorm_dict, rv, lv  = self.base_tnorm(args, function, probs)
        return {
            **tnorm_dict,
            torch.logical_and: lambda: lv * rv,
            torch.logical_or: lambda: lv + rv - lv * rv,
        }[function]()

class LukasiewiczTNormSolver(TNormSolver):
    def gettnorm(self, args, function, probs):
        tnorm_dict, rv, lv = self.base_tnorm(args, function, probs)
        return {
            **tnorm_dict,
            torch.logical_and: lambda: torch.relu(lv + rv - 1),
            torch.logical_or: lambda: 1 - torch.relu(1 - lv - rv),
        }[function]()


class GodelTNormSolver(TNormSolver):
    def gettnorm(self, args, function, probs):
        tnorm_dict, rv, lv = self.base_tnorm(args, function, probs)
        return {
            **tnorm_dict,
            torch.logical_and: lambda: torch.min(lv,rv),
            torch.logical_or: lambda: torch.max(lv, rv),
        }[function]()


class LukasiewiczTNormSolver(TNormSolver):
    def gettnorm(self, lv, l_type, rv, r_type, function):
        return {
            torch.logical_and: lambda: torch.relu(lv + rv - 1),
            torch.logical_or: lambda: 1 - torch.relu(1 - lv - rv),
            torch.ne: lambda: (1 - lv * rv).sum(dim=-1),
            torch.Tensor.__getitem__: lambda: lv[rv],
        }[function]()


class GodelTNormSolver(TNormSolver):
    def gettnorm(self, lv, l_type, rv, r_type, function):
        return {
            torch.logical_and: lambda: torch.min(lv,rv),
            torch.logical_or: lambda: torch.max(lv, rv),
            torch.ne: lambda: (1 - lv * rv).sum(dim=-1),
            torch.Tensor.__getitem__: lambda: lv[rv],
        }[function]()

