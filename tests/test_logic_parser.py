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
def oper_not(y): return not y[0]
def oper_and(y): return y[0] and y[1]
def oper_or(y): return y[0] or y[1]
def attr_not(y): return y[0].logical_not()
def attr_and(y): return y[0].logical_and(y[1])
def attr_or(y): return y[0].logical_or(y[1])


def two_vars(x, y): return x[0] == y[1]
def x_implies_y(x, y): return x[0] <= y[1]


def two_vars_assign(x, y):
    v = x[0] == y[1]
    return v


def forall_list(y): return all([y[0]])
def exists_list(y): return any([y[0]])


def forall_var(y): return all(y)
def exists_var(y): return any(y)


def forall_cond(x, y): return all(y[x == 2])


@pytest.fixture
def parses():
    return [
        (true, Const(True)),
        (false, Const(False)),
        (const0, Const(False)),
        (const1, Const(True)),
        (equals1_2, IsEq(Const(1), Const(2))),
        (y_eq_const, IsEq(VarList(Arg('y', 0), [0]), Const(0))),
        (const_eq_y, IsEq(Const(0), VarList(Arg('y', 0), [0]))),
        (var_eq_var, IsEq(VarList(Arg('y', 0), [0]), VarList(Arg('y', 0), [1]))),
        (y0, VarList(Arg('y', 0), [0])),
        (oper_not, Not(VarList(Arg('y', 0), [0]))),
        (attr_not, Not(VarList(Arg('y', 0), [0]))),
        (oper_and, And(VarList(Arg('y', 0), [0]), VarList(Arg('y', 0), [1]))),
        (attr_and, And(VarList(Arg('y', 0), [0]), VarList(Arg('y', 0), [1]))),
        (oper_or, Or(VarList(Arg('y', 0), [0]), VarList(Arg('y', 0), [1]))),
        (attr_or, Or(VarList(Arg('y', 0), [0]), VarList(Arg('y', 0), [1]))),
        (two_vars, IsEq(VarList(Arg('x', 0), [0]), VarList(Arg('y', 1), [1]))),
        (x_implies_y, Implication(VarList(Arg('x', 0), [0]),
                                  VarList(Arg('y', 1), [1]))),
        (forall_list, Forall(List([VarList(Arg('y', 0), [0])]))),
        (exists_list, Exists(List([VarList(Arg('y', 0), [0])]))),
        (forall_var, Forall(Arg('y', 0))),
        (exists_var, Exists(Arg('y', 0)))
    ]


def test_parses(parses):
    finder = FunDefFindingVisitor()
    for (f, tree) in parses:
        astree = parse_object(f)
        fundef = finder.visit(astree)
        parser = LogicExpressionASTVisitor()
        ptree = parser.visit(fundef)
        print(ptree, tree)
        assert ptree.return_node == tree
