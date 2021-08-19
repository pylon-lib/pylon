from typing import Tuple
import pytest

from pytorch_constraints.lazy_tensor import *
from pytorch_constraints.constraint import constraint
from pytorch_constraints.shaped_lazy_solver import ProductTNormSolver

def rand_tensor(dims, isBool=True):
    if isBool:
        return torch.rand(dims) < .5
    return torch.rand(dims)


def parse_pn(pn) -> Tuple[LazyTensor, torch.tensor]:
    """Convert a postfix notation of tensor expressions to its equivalent LazyTensor object and torch.Tensor object"""
    if len(pn) > 0:
        lt_stack = []
        rt_stack = []
        i = 0
        while i < len(pn):
            if not callable(pn[i]):
                if isinstance(pn[i], torch.Tensor):
                    lt_item = ConstLazyTensor(pn[i])
                else:
                    lt_item = pn[i]
                rt_item = pn[i]
            elif pn[i] == torch.logical_not:
                lt_item = new_lazy_tensor(torch.logical_not, [lt_stack.pop()])
                rt_item = pn[i](rt_stack.pop())
            else:
                lt_item = new_lazy_tensor(pn[i], [lt_stack.pop(), lt_stack.pop()])
                rt_item = pn[i](rt_stack.pop(), rt_stack.pop())

            lt_stack.append(lt_item)
            rt_stack.append(rt_item)
            i += 1

        return lt_stack.pop(), rt_stack.pop()

    return None, None


@pytest.fixture
def lazy_tensor_real_tensor_pairs():
    """ Return a list of tuples (parsed tensor of LazyTensor, expected Tensor, errMsg) """
    err_msg_fmt = "LazyTensor {} failed"
    two_dims = (10, 10)

    # single
    not_pn = [rand_tensor(two_dims), torch.logical_not]
    and_pn = [rand_tensor(two_dims), rand_tensor(two_dims), torch.logical_and]
    or_pn = [rand_tensor(two_dims), rand_tensor(two_dims), torch.logical_or]

    returned = [
        (*parse_pn(not_pn), err_msg_fmt.format("logical_not")),
        (*parse_pn(and_pn), err_msg_fmt.format("logical_and")),
        (*parse_pn(or_pn), err_msg_fmt.format("logical_or"))
    ]

    # 2-level nested
    or_nested_pn = [*and_pn, *or_pn, torch.logical_or]
    returned.append((*parse_pn(or_nested_pn),
                     err_msg_fmt.format("2-level nested with logical_or as the top operator")))
    and_nested_pn = [*and_pn, *or_pn, torch.logical_and]
    returned.append((*parse_pn(and_nested_pn),
                     err_msg_fmt.format("2-level nested with logical_and as the top operator")))

    # 3-level nested
    nested_pn = [*or_nested_pn, *and_nested_pn, torch.logical_and]
    returned.append((*parse_pn(nested_pn),
                     err_msg_fmt.format("3-level nested with logical_and as the top operator")))

    # place __getitem__'s key first since the postfix notation stack is popped later
    # then the key is placed after the tensor object
    returned.append((*parse_pn([2, *nested_pn, torch.Tensor.__getitem__])
                     , err_msg_fmt.format("[2]")))
    # len(key) == 2
    returned.append((*parse_pn([(slice(None, None, None), 2), *nested_pn, torch.Tensor.__getitem__])
                     , err_msg_fmt.format("[:,2]")))
    returned.append((*parse_pn([(2, slice(None, None, None)), *nested_pn, torch.Tensor.__getitem__])
                     , err_msg_fmt.format("[2,:]")))
    returned.append((*parse_pn([(slice(None, 2, None), 2), *nested_pn, torch.Tensor.__getitem__])
                     , err_msg_fmt.format("[:2,2]")))
    returned.append((*parse_pn([(slice(2, None, None), 2), *nested_pn, torch.Tensor.__getitem__])
                     , err_msg_fmt.format("[2:,2]")))
    returned.append((*parse_pn([(slice(2, 7, None), 2), *nested_pn, torch.Tensor.__getitem__])
                     , err_msg_fmt.format("[2:7,2]")))
    returned.append((*parse_pn([(slice(2, 7, 2), 2), *nested_pn, torch.Tensor.__getitem__])
                     , err_msg_fmt.format("[2:7:2,2]")))
    returned.append((*parse_pn([(slice(2, 7, 2), slice(1, 9, 3)), *nested_pn, torch.Tensor.__getitem__])
                     , err_msg_fmt.format("[full slice]*2 [2:7:2,1:9:3]")))
    # len(key) == 3
    three_dims = (10, 10, 10)
    and_pn = [rand_tensor(three_dims), rand_tensor(three_dims), torch.logical_and]
    returned.append((*parse_pn([(1, 2, 3), *and_pn, torch.Tensor.__getitem__])
                     , err_msg_fmt.format("[1,2,3]")))
    returned.append((*parse_pn([(1, slice(2, None, None), 3), *and_pn, torch.Tensor.__getitem__])
                     , err_msg_fmt.format("[1,2:,3]")))
    returned.append((*parse_pn([(slice(2, 7, 2), slice(1, 9, 3), slice(2, 7, 2)), *and_pn, torch.Tensor.__getitem__])
                     , err_msg_fmt.format("[full slice]*3 [2:7:2, 1:9:3, 2:7:2]")))

    # len(key) == 4
    three_dims = (10, 10, 10, 10)
    and_pn = [rand_tensor(three_dims), rand_tensor(three_dims), torch.logical_and]
    returned.append((*parse_pn([(1, 2, 3), *and_pn, torch.Tensor.__getitem__])
                     , err_msg_fmt.format("[1,2,3,4]")))
    returned.append((*parse_pn([(1, slice(2, None, None), 3, slice(None, 4, None)), *and_pn, torch.Tensor.__getitem__])
                     , err_msg_fmt.format("[1,2:,3,:4]")))
    returned.append((*parse_pn([(slice(2, 7, 2), slice(1, 9, 3), slice(2, 7, 2), slice(1, 9, 3)), *and_pn, torch.Tensor.__getitem__])
                     , err_msg_fmt.format("[full slice]*4 [2:7:2, 1:9:3, 2:7:2, 1:9:3]")))

    # apply logical gate after slicing
    sliced_pn = [(slice(2, 7, 2), slice(1, 9, 3)), *nested_pn, torch.Tensor.__getitem__]
    returned.append((*parse_pn([*sliced_pn, torch.logical_not])
                     , err_msg_fmt.format("logical_not after __getitem__(key=[full slice, full slice])")))
    returned.append((*parse_pn([*sliced_pn, *sliced_pn, torch.logical_and])
                     , err_msg_fmt.format("logical_and after __getitem__(key=[full slice, full slice])")))
    returned.append((*parse_pn([*sliced_pn, *sliced_pn, torch.logical_or])
                     , err_msg_fmt.format("logical_or after __getitem__(key=[full slice, full slice])")))

    # apply ==
    returned.append((*parse_pn([True, *sliced_pn, torch.eq])
                     , err_msg_fmt.format("== single value")))
    returned.append((*parse_pn([*sliced_pn, *sliced_pn, torch.eq])
                     , err_msg_fmt.format("== other lazy tensor")))

    # apply calculations
    pn = [rand_tensor(two_dims, isBool=False)]
    returned.append((*parse_pn([-1, *pn, torch.sum])
                     , err_msg_fmt.format("sum(dim=-1)")))
    returned.append((*parse_pn([-1, *pn, torch.nn.functional.softmax])
                     , err_msg_fmt.format("softmax(dim=-1)")))

    return returned


def test_lazy_tensor_to_tensor(lazy_tensor_real_tensor_pairs):
    for (lazy_tensor, tensor, err_msg) in lazy_tensor_real_tensor_pairs:
        assert torch.equal(lazy_tensor.tensor(), tensor), err_msg

def test_lazy_solver_equals():
    def equality_test(a):
        return (a[:,2] == 1)
    cons = constraint(equality_test, ProductTNormSolver())
    tensor1 = torch.randn((3, 2))
    constraint_loss = cons(tensor1)
    assert True
