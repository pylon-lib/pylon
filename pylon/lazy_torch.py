import torch
from .lazy_tensor import new_lazy_tensor


def ones(*shape):
	return new_lazy_tensor(torch.ones, (None, shape))

def zeros(*shape):
	return new_lazy_tensor(torch.zeros, (None, shape))

def eye(length):
	return new_lazy_tensor(torch.eye, (None, length))

def cat(tensors, dim):
	return new_lazy_tensor(torch.cat, (None, tensors, dim))

def stack(tensors, dim):
	return new_lazy_tensor(torch.stack, (None, tensors, dim))

def tile(tensors, shape):
    return new_lazy_tensor(torch.tile, (None, tensors, shape))

def randn(*shape):
	return new_lazy_tensor(torch.randn, (None, shape))

def logical_and(left, right):
	return new_lazy_tensor(torch.logical_and, (None, left, right))

def logical_or(left, right):
	return new_lazy_tensor(torch.logical_or, (None, left, right))

def logical_not(left):
	return new_lazy_tensor(torch.logical_xor, (None, left))

def sum(tensor, dim):
	return new_lazy_tensor(torch.sum, (None, tensor, dim))

def all(tensor, dim=None):
	return new_lazy_tensor(torch.all, (None, tensor, dim))

def any(tensor, dim=None):
	return new_lazy_tensor(torch.any, (None, tensor, dim))

def squeeze(tensor, dim):
	return new_lazy_tensor(torch.squeeze, (None, tensor, dim))

def unsqueeze(tensor, dim):
	return new_lazy_tensor(torch.unsqueeze, (None, tensor, dim))

def add(left, right):
	return new_lazy_tensor(torch.add, (None, left, right))

def sub(left, right):
	return new_lazy_tensor(torch.sub, (None, left, right))

def mul(left, right):
	return new_lazy_tensor(torch.mul, (None, left, right))

def div(left, right):
	return new_lazy_tensor(torch.div, (None, left, right))

def floor_divide(left, right):
	return new_lazy_tensor(torch.floor_divide, (None, left, right))	

def sqrt(tensor):
	return new_lazy_tensor(torch.sqrt, (None, tensor))

def rsqrt(tensor):
	return new_lazy_tensor(torch.rsqrt, (None, tensor))

def mm(left, right):
	return new_lazy_tensor(torch.mm, (None, left, right))

def bmm(left, right):
	return new_lazy_tensor(torch.bmm, (None, left, right))

def log(tensor):
	return new_lazy_tensor(torch.log, (None, tensor))

def exp(tensor):
	return new_lazy_tensor(torch.exp, (None, tensor))

def relu(tensor):
	return new_lazy_tensor(torch.relu, (None, tensor))

def sigmoid(tensor):
	return new_lazy_tensor(torch.sigmoid, (None, tensor))

def softmax(tensor, dim):
	return new_lazy_tensor(torch.softmax, (None, tensor, dim))

def logsumexp(tensor, dim):
	return new_lazy_tensor(torch.logsumexp, (None, tensor, dim))

def masked_select(tensor, mask):
	return new_lazy_tensor(torch.masked_select, (None, tensor, mask))

# TODO, add other interfaces like gather