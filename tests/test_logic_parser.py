import pytest
import ast
import inspect

from pytorch_constraints.ast_visitor import *


def parse_object(obj):
    source = inspect.getsource(obj)
    return ast.parse(source)


def true(): return True
def false(): return False
def const0(): return 0
def const1(): return 1
def equals1_2(): return 1 == 2
def y0(): return y[0]
def y_eq_const(): return y[0] == 0


@pytest.fixture
def parses():
    return [
        (true, Const(True)),
        (false, Const(False)),
        (const0, Const(False)),
        (const1, Const(True)),
        (equals1_2, IsEq(Const(1), Const(2))),
        (y0, Not(IsEq(VarUse(0), Const(0)))),
        (y_eq_const, IsEq(VarUse(0), Const(0)))
    ]


def test_parses(parses):
    finder = FunDefFindingVisitor()
    parser = LogicExpressionVisitor()
    for (f, tree) in parses:
        astree = parse_object(f)
        ptree = parser.visit(finder.visit(astree))
        print(ptree)
        assert ptree == tree
