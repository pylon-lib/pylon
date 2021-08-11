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

# TODO, add other interfaces like gather