import ast
from functools import reduce
import torch
from .tree_node import *


class FunDefFindingVisitor(ast.NodeVisitor):
    def __init__(self):
        super(ast.NodeVisitor).__init__()

    def visit_Module(self, node):
        return self.visit(node.body[0])

    def visit_Assign(self, node):
        return self.visit(node.value)

    def visit_Call(self, node):
        assert(node.func.id == 'constraint')
        return self.visit(node.args[0])

    def visit_Lambda(self, node):
        return node

    def visit_FunctionDef(self, node):
        return node


class LogicExpressionASTVisitor(ast.NodeVisitor):

    def __init__(self):
        self.arg_pos = {}
        self.iddefs = {}
        super(ast.NodeVisitor).__init__()

    def generic_visit(self, node):
        print(ast.dump(node))
        raise NotImplementedError

    def get_arg_pos(self, node):
        arg_pos = {}
        for i, arg in enumerate(node.args.args):
            arg_pos[arg.arg] = i
        return arg_pos

    def visit_FunctionDef(self, node):
        self.arg_pos = self.get_arg_pos(node)
        for b in node.body:
            body_tree = self.visit(b)
        return FunDef(self.arg_pos, self.iddefs, body_tree.as_bool())

    def visit_Lambda(self, node):
        self.arg_pos = self.get_arg_pos(node)
        body_tree = self.visit(node.body)
        return FunDef(self.arg_pos, self.iddefs, body_tree.as_bool())

    def visit_Return(self, node):
        return self.visit(node.value).as_bool()

    def visit_UnaryOp(self, node):
        supported = {
            ast.Not: (lambda opr: Not(opr.as_bool()))
        }
        op_func = supported[type(node.op)]
        opr = self.visit(node.operand)
        return op_func(opr)

    def visit_Name(self, node):
        # identify whether it is a local variable or an argument
        if node.id in self.iddefs:
            # assumes unscripted reference is to local variable
            iddef = self.iddefs[node.id]
            return IdentifierRef(iddef)
        else:
            assert node.id in self.arg_pos
            arg_name = node.id
            arg_pos = self.arg_pos[node.id]
            return Arg(arg_name, arg_pos)

    def visit_Subscript(self, node):
        arg = self.visit(node.value)
        select = self.visit(node.slice.value)
        return Subscript(arg, select)

    def visit_Assign(self, node):
        assert len(node.targets) == 1
        id = node.targets[0].id
        definition = self.visit(node.value)
        iddef = IdentifierDef(id, definition)
        assert id not in self.iddefs
        self.iddefs[id] = iddef
        return iddef

    def visit_List(self, node):
        elts = [self.visit(elt) for elt in node.elts]
        print(elts)
        return List(elts)

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name):
            fname = node.func.id
            if fname == 'all':
                return Forall(self.visit(node.args[0]))
            if fname == 'any':
                return Exists(self.visit(node.args[0]))
            if fname == 'all':
                return Forall(self.visit(node.args[0]))
        elif isinstance(node.func, ast.Attribute):
            fname = node.func.attr
            if fname == 'logical_not':
                return Not(self.visit(node.func.value))
            if fname == 'logical_and':
                return And(self.visit(node.func.value), self.visit(node.args[0]))
            if fname == 'logical_or':
                return Or(self.visit(node.func.value), self.visit(node.args[0]))
        raise NotImplementedError(node)

    def visit_NameConstant(self, node):
        #deprecated in 3.8
        return Const(node.value)

    def visit_Num(self, node):
        #deprecated in 3.8
        return Const(node.n)

    def visit_Constant(self, node):
        return Const(node.value)

    def visit_BoolOp(self, node):
        supported = {
            ast.And: (lambda left, right: And(left.as_bool(), right.as_bool())),
            ast.Or: (lambda left, right: Or(left.as_bool(), right.as_bool()))
        }
        op_func = supported[type(node.op)]
        trees = map(self.visit, node.values)
        return reduce(op_func, trees)

    def visit_Compare(self, node):
        supported = {
            ast.Eq: (lambda left, right: IsEq(left, right)),
            ast.NotEq: (lambda left, right: Not(IsEq(left, right))),
            ast.LtE:  (lambda left, right: Implication(left.as_bool(), right.as_bool()))   # implication/residuum
        }
        assert(len(node.ops))
        op_func = supported[type(node.ops[0])]
        ltree = self.visit(node.left)
        assert(len(node.comparators))
        rtree = self.visit(node.comparators[0])
        return op_func(ltree, rtree)
