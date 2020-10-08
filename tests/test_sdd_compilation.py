import pytest
import ast
import inspect

from pytorch_constraints.ast_visitor import *
from pytorch_constraints.circuit_solver import SddVisitor
from pysdd.sdd import SddManager, Vtree, WmcManager


def parse_object(obj):
    finder = FunDefFindingVisitor()
    source = inspect.getsource(obj)
    astree = ast.parse(source)
    fundef = finder.visit(astree)
    parser = LogicExpressionASTVisitor(source, obj.__globals__)
    return parser.visit(fundef)


def compile(obj):
    ptree = parse_object(obj)
    vtree = Vtree(var_count=1)
    mgr = SddManager.from_vtree(vtree)
    return SddVisitor().visit(ptree, mgr)


def true(y): return True
def false(y): return False
def const0(y): return 0
def const1(y): return 1
def equals1_2(y): return 1 == 2
def y_eq_const1(y): return y[0] == 0
def y_eq_const2(y): return y[0] == True
def const_eq_y(y): return 0 == y[0]
def var_eq_var(y): return y[0] == y[1]
def y0(y): return y[0]


def test_true():
    sdd = compile(true)
    assert sdd.is_true()
    sdd = compile(false)
    assert sdd.is_false()
    sdd = compile(y_eq_const1)
    assert sdd.is_literal()
    sdd = compile(y_eq_const2)
    assert sdd.is_literal()
    sdd = compile(const_eq_y)
    assert sdd.is_literal()
    sdd = compile(var_eq_var)
    assert sdd.size() > 1
    pass
