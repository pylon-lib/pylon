import torch
from abc import ABC, abstractmethod


def new_lazy_tensor(function, args):
    return LazyTensor(function, args)


class AbstractLazyTensor(ABC):
    def logical_not(self):
        return new_lazy_tensor(torch.Tensor.logical_not, [self])

    def logical_and(self, arg):
        return new_lazy_tensor(torch.Tensor.logical_and, [self, arg])

    def logical_or(self, arg):
        return new_lazy_tensor(torch.Tensor.logical_or, [self, arg])

    def logical_xor(self, arg):
        return new_lazy_tensor(torch.Tensor.logical_xor, [self, arg])

    def sum(self, dim=None):
        return new_lazy_tensor(torch.Tensor.sum, (self, dim))

    def softmax(self, dim):
        return new_lazy_tensor(torch.Tensor.softmax, (self, dim))

    def sigmoid(self):
        return new_lazy_tensor(torch.Tensor.sigmoid, (self))

    def relu(self):
        return new_lazy_tensor(torch.Tensor.relu, (self))

    def min(self, dim=None):
        return new_lazy_tensor(torch.Tensor.min, (self, dim))

    def max(self, dim=None):
        return new_lazy_tensor(torch.Tensor.max, (self, dim))

    def all(self, dim):
        return new_lazy_tensor(torch.Tensor.all, (self, dim))

    def any(self, dim):
        return new_lazy_tensor(torch.Tensor.any, (self, dim))

    def squeeze(self, dim):
        return new_lazy_tensor(torch.Tensor.squeeze, (self,))

    def unsqueeze(self, dim):
        return new_lazy_tensor(torch.Tensor.unsqueeze, (self, dim))

    def clone(self):
        return new_lazy_tensor(torch.clone, (self, None))

    def expand_as(self, arg):
        return new_lazy_tensor(torch.Tensor.expand_as, (self, arg))

    def size(self):
        return new_lazy_tensor(torch.Tensor.size, (self,))

    def diagonal(self, offset, dim1, dim2):
        return new_lazy_tensor(torch.Tensor.diagonal, (self, offset, dim1, dim2))

    def zero_(self):
        raise Exception('in-place operation not supported.')

    def add_(self, arg):
        raise Exception('in-place operation not supported.')

    def sub_(self, arg):
        raise Exception('in-place operation not supported.')

    def mul_(self, arg):
        raise Exception('in-place operation not supported.')

    def div_(self, arg):
        raise Exception('in-place operation not supported.')

    def tile(self, *arg):
        return new_lazy_tensor(torch.Tensor.tile, (self, *arg))

    def add(self, arg):
        return new_lazy_tensor(torch.Tensor.add, (self, arg))

    def sub(self, arg):
        return new_lazy_tensor(torch.Tensor.sub, (self, arg))

    def mul(self, arg):
        return new_lazy_tensor(torch.Tensor.multiply, (self, arg))

    def div(self, arg):
        return new_lazy_tensor(torch.Tensor.div, (self, arg))

    def bmm(self, arg):
        return new_lazy_tensor(torch.Tensor.bmm, (self, arg))

    def mm(self, arg):
        return new_lazy_tensor(torch.Tensor.mm, (self, arg))

    def log(self):
        return new_lazy_tensor(torch.Tensor.log, [self])

    def exp(self):
        return new_lazy_tensor(torch.Tensor.exp, [self])

    def logsumexp(self, arg):
        return new_lazy_tensor(torch.Tensor.logsumexp, [self, arg])

    def masked_select(self, arg):
        return new_lazy_tensor(torch.Tensor.masked_select, [self, arg])        

    def float(self):
        return new_lazy_tensor(torch.Tensor.float, [self])

    def half(self):
        return new_lazy_tensor(torch.Tensor.half, [self])

    def double(self):
        return new_lazy_tensor(torch.Tensor.double, [self])

    def int(self):
        return new_lazy_tensor(torch.Tensor.int, [self])

    def short(self):
        return new_lazy_tensor(torch.Tensor.short, [self])

    def long(self):
        return new_lazy_tensor(torch.Tensor.long, [self])

    def byte(self):
        return new_lazy_tensor(torch.Tensor.byte, [self])

    def bool(self):
        return new_lazy_tensor(torch.Tensor.bool, [self])

    def __getitem__(self, key):
        return new_lazy_tensor(torch.Tensor.__getitem__, [self, key])

    def __gt__(self, other):
        return new_lazy_tensor(torch.gt, [self, other])

    def __lt__(self, other):
        return new_lazy_tensor(torch.lt, [self, other])

    def __ne__(self, other):
        return new_lazy_tensor(torch.ne, [self, other])

    def __eq__(self, other):
        return new_lazy_tensor(torch.eq, [self, other])

    def __add__(self, other):
        return new_lazy_tensor(torch.Tensor.add, (self, other))

    def __sub__(self, other):
        return new_lazy_tensor(torch.Tensor.sub, (self, other))

    def __mul__(self, other):
        return new_lazy_tensor(torch.Tensor.mul, (self, other))

    def __truediv__(self, other):
        return new_lazy_tensor(torch.Tensor.div, (self, other))

    def __floordiv__(self, other):
        return new_lazy_tensor(torch.Tensor.floor_divide, (self, other))

    def __mod__(self, other):
        return new_lazy_tensor(torch.Tensor.remainder, (self, other))

    def __lshift__(self, other):
        raise NotImplemented
        # return new_lazy_tensor(torch., (self, other))

    def __rshift__(self, other):
        raise NotImplemented
        # return new_lazy_tensor(torch., (self, other))

    def __and__(self, other):
        return new_lazy_tensor(torch.Tensor.bitwise_and, (self, other))

    def __or__(self, other):
        return new_lazy_tensor(torch.Tensor.bitwise_or, (self, other))

    def __xor__(self, other):
        return new_lazy_tensor(torch.Tensor.bitwise_xor, (self, other))

    def __invert__(self, other):
        return new_lazy_tensor(torch.Tensor.bitwise_not, (self, other))

    def __le__(self, other):
        return new_lazy_tensor(torch.le, (self, other))

    def __ge__(self, other):
        return new_lazy_tensor(torch.le, (other, self))

    @abstractmethod
    def __str__(self, level=0):
        print("Some implementation!")


class ConstShapedLazyTensor(AbstractLazyTensor):
    def __init__(self, index):
        self.index = index
        # self.shape = shape

    def __str__(self, level=0):
        return "  " * level + f'ConstShapedLazyTensor: index={self.index}'


class ConstLazyTensor(AbstractLazyTensor):
    def __init__(self, value: torch.Tensor):
        self.value = value

    def tensor(self):
        return self.value

    def __str__(self, level=0):
        return "  " * level + f'ConstLazyTensor: value=\n{self.value}'


class LazyTensor(AbstractLazyTensor):
    def __init__(self, function, args):
        self.function = function
        self.args = args

    def tensor(self):
        tensor_args = []
        for arg in self.args:
            if issubclass(arg.__class__, AbstractLazyTensor):
                tensor_args.append(arg.tensor())
            else:
                tensor_args.append(arg)
        return self.function(*tensor_args)

    def __str__(self, level=0):
        ret = "  " * level + "LazyTensor:" + self.function.__name__ + "\n"
        for arg in self.args:
            if issubclass(arg.__class__, AbstractLazyTensor):
                ret += arg.__str__(level + 1) + "\n"
            elif arg == Ellipsis:  # 3 dots
                ret += "  " * (level + 1) + "...\n"
            else:
                ret += "  " * (level + 1) + str(arg) + "\n"
        return ret
