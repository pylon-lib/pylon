import torch
from abc import ABC, abstractmethod


def new_lazy_tensor(function, args):
    return LazyTensor(function, args)


class AbstractLazyTensor(ABC):
    def logical_not(self):
        return new_lazy_tensor(torch.logical_not, [self])

    def logical_and(self, arg):
        return new_lazy_tensor(torch.logical_and, [self, arg])

    def logical_or(self, arg):
        return new_lazy_tensor(torch.logical_or, [self, arg])

    def logical_xor(self, arg):
        return new_lazy_tensor(torch.logical_xor, [self, arg])

    def all(self):
        return new_lazy_tensor(torch.all, [self])

    def sum(self, dim):
        return new_lazy_tensor(torch.sum, (self, dim))

    def softmax(self, dim):
        return new_lazy_tensor(torch.nn.functional.softmax, (self, dim))

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
        return new_lazy_tensor(torch.add, (self, other))

    def __sub__(self, other):
        return new_lazy_tensor(torch.sub, (self, other))

    def __mul__(self, other):
        return new_lazy_tensor(torch.mul, (self, other))

    def __truediv__(self, other):
        return new_lazy_tensor(torch.div, (self, other))

    def __floordiv__(self, other):
        return new_lazy_tensor(torch.floor_divide, (self, other))

    def __mod__(self, other):
        return new_lazy_tensor(torch.remainder, (self, other))

    def __lshift__(self, other):
        raise NotImplemented
        # return new_lazy_tensor(torch., (self, other))

    def __rshift__(self, other):
        raise NotImplemented
        # return new_lazy_tensor(torch., (self, other))

    def __and__(self, other):
        return new_lazy_tensor(torch.bitwise_and, (self, other))

    def __or__(self, other):
        return new_lazy_tensor(torch.bitwise_or, (self, other))

    def __xor__(self, other):
        return new_lazy_tensor(torch.bitwise_xor, (self, other))

    def __invert__(self, other):
        return new_lazy_tensor(torch.bitwise_not, (self, other))

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
