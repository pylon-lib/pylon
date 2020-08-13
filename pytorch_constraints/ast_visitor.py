import ast
from functools import reduce


class FunDefFindingVisitor(ast.NodeVisitor):
    def __init__(self):
        super(ast.NodeVisitor).__init__()

    def get_arg_pos(self, node):
        arg_pos = {}
        for i, arg in enumerate(node.args.args):
            arg_pos[arg.arg] = i
        print(arg_pos)
        return arg_pos

    def visit_Module(self, node):
        return self.visit(node.body[0])

    def visit_Assign(self, node):
        return self.visit(node.value)

    def visit_Call(self, node):
        assert(node.func.id == 'constraint')
        return self.visit(node.args[0])

    def visit_Lambda(self, node):
        return node, self.get_arg_pos(node)

    def visit_FunctionDef(self, node):
        return node, self.get_arg_pos(node)


class TreeNode:
    def __init__(self, name, children):
        self.name = name
        self.children = children

    def __str__(self):
        arg_str = "(" + ",".join([str(c) for c in self.children]) + ")" if len(self.children) > 0 else ""
        return self.name + arg_str

    def as_bool(self):
        return self

    def prod_tnorm(self, probs):
        print(self)
        raise NotImplementedError

    def __eq__(self, obj):
        return (type(obj) == type(self) and
                self.name == obj.name and self.children == obj.children)


class BinaryOp(TreeNode):
    def __init__(self, name, left, right):
        self.left = left
        self.right = right
        super().__init__(name, [left, right])


class And(BinaryOp):
    def __init__(self, left, right):
        super().__init__("And", left, right)

    def prod_tnorm(self, probs):
        lv = self.left.prod_tnorm(probs)
        rv = self.right.prod_tnorm(probs)
        return lv * rv


class Or(BinaryOp):
    def __init__(self, left, right):
        super().__init__("Or", left, right)

    def prod_tnorm(self, probs):
        lv = self.left.prod_tnorm(probs)
        rv = self.right.prod_tnorm(probs)
        return lv + rv - lv * rv


class UnaryOp(TreeNode):
    def __init__(self, name, operand):
        self.operand = operand
        super().__init__(name, [operand])


class Not(UnaryOp):
    def __init__(self, operand):
        super().__init__("Not", operand)

    def prod_tnorm(self, probs):
        return 1.0 - self.operand.prod_tnorm(probs)


class IsEq(BinaryOp):
    def __init__(self, left, right):
        super().__init__('Eq', left, right)

    def prod_tnorm(self, probs):
        if isinstance(self.left, VarUse) and isinstance(self.right, Const):
            return self.left.probs(probs)[self.right.value]
        elif isinstance(self.left, Const) and isinstance(self.right, VarUse):
            return self.right.probs(probs)[self.left.value]
        elif isinstance(self.left, Const) and isinstance(self.right, Const):
            return 1.0 if self.left.value == self.right.value else 0.0
        elif isinstance(self.left, VarUse) and isinstance(self.right, VarUse):
            return (self.left.probs(probs)*self.right.probs(probs)).sum()
        else:
            raise NotImplementedError


class Const(TreeNode):
    def __init__(self, value):
        self.value = value
        self.is_bool = isinstance(value, bool)
        super().__init__(str(value), [])

    def as_bool(self):
        return self if self.is_bool else Const(bool(self.value))

    def prod_tnorm(self, probs):
        return 1.0 if self.value == self.value else 0.0


class VarUse(TreeNode):
    def __init__(self, varidx, varname, index):
        self.varidx = varidx
        self.varname = varname
        self.index = index
        super().__init__(self.varname + '[' + str(self.index) + "]", [])

    def probs(self, probs):
        return probs[self.varidx][self.index]

    def as_bool(self):
        return Not(IsEq(self, Const(0)))


class LogicExpressionVisitor(ast.NodeVisitor):

    def __init__(self, arg_pos):
        self.arg_pos = arg_pos
        super(ast.NodeVisitor).__init__()

    def generic_visit(self, node):
        print(ast.dump(node))
        raise NotImplementedError

    def visit_FunctionDef(self, node):
        # TODO: handle multiple lines, and do something with arguments?
        body_tree = self.visit(node.body[0])
        return body_tree

    def visit_Lambda(self, node):
        # Same as FunctionDef?
        body_tree = self.visit(node.body)
        return body_tree.as_bool()

    def visit_Return(self, node):
        return self.visit(node.value).as_bool()

    def visit_UnaryOp(self, node):
        supported = {
            ast.Not: (lambda opr: Not(opr.as_bool()))
        }
        op_func = supported[type(node.op)]
        opr = self.visit(node.operand)
        return op_func(opr)

    def visit_Subscript(self, node):
        # TODO check node.value is the variable?
        varidx = self.arg_pos[node.value.id]
        return VarUse(varidx, node.value.id, node.slice.value.n)

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
            ast.NotEq: (lambda left, right: Not(IsEq(left, right)))
        }
        assert(len(node.ops))
        op_func = supported[type(node.ops[0])]
        ltree = self.visit(node.left)
        assert(len(node.comparators))
        rtree = self.visit(node.comparators[0])
        return op_func(ltree, rtree)
