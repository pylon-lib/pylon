import pytest
import ast
import inspect

from pytorch_constraints.ast_visitor import *


def parse_object(obj):
    source = inspect.getsource(obj)
    return ast.parse(source)


def true(y): return True
def false(y): return False
def const0(y): return 0
def const1(y): return 1
def equals1_2(y): return 1 == 2
def y_eq_const(y): return y[0] == 0
def const_eq_y(y): return 0 == y[0]
def var_eq_var(y): return y[0] == y[1]
def y0(y): return y[0]


def two_vars(x, y): return x[0] == y[1]


def two_vars_assign(x, y):
    v = x[0] == y[1]
    return v


@pytest.fixture
def parses():
    return [
        (true, Const(True)),
        (false, Const(False)),
        (const0, Const(False)),
        (const1, Const(True)),
        (equals1_2, IsEq(Const(1), Const(2))),
        (y_eq_const, IsEq(VarUse(0, 'y', 0), Const(0))),
        (const_eq_y, IsEq(Const(0), VarUse(0, 'y', 0))),
        (var_eq_var, IsEq(VarUse(0, 'y', 0), VarUse(0, 'y', 1))),
        (y0, Not(IsEq(VarUse(0, 'y', 0), Const(0)))),
        (two_vars, IsEq(VarUse(0, 'x', 0), VarUse(1, 'y', 1)))
    ]


def test_parses(parses):
    finder = FunDefFindingVisitor()
    for (f, tree) in parses:
        astree = parse_object(f)
        fundef = finder.visit(astree)
        parser = LogicExpressionVisitor()
        ptree = parser.visit(fundef)
        print(ptree)
        assert ptree.return_node == tree
