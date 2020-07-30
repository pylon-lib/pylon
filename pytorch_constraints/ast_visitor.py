import ast


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


class TreeNode:
    def __init__(self, name, children):
        self.name = name
        self.children = children

    def __str__(self):
        arg_str = "(" + ",".join([str(c) for c in self.children]) + ")" if len(self.children) > 0 else ""
        return self.name + arg_str


class VariableUse(TreeNode):
    def __init__(self, index):
        self.index = index
        super().__init__('y[' + str(self.index) + "]", [])


class BinaryOp(TreeNode):
    def __init__(self, name, left, right):
        self.left = left
        self.right = right
        super().__init__(name, [left, right])


class And(BinaryOp):
    def __init__(self, left, right):
        super().__init__("And", left, right)


class Or(BinaryOp):
    def __init__(self, left, right):
        super().__init__("Or", left, right)


class UnaryOp(TreeNode):
    def __init__(self, name, operand):
        self.operand = operand
        super().__init__(name, [operand])


class Not(UnaryOp):
    def __init__(self, operand):
        super().__init__("Not", operand)


class LogicExpressionVisitor(ast.NodeVisitor):

    def __init__(self):
        super(ast.NodeVisitor).__init__()

    def generic_visit(self, node):
        print(ast.dump(node))
        raise NotImplementedError

    # def visit_Module(self, node):
    #     # TODO: assume only once, and single line
    #     return self.visit(node.body[0])

    def visit_FunctionDef(self, node):
        # TODO: handle multiple lines, and do something with arguments?
        body_tree = self.visit(node.body[0])
        return body_tree

    def visit_Lambda(self, node):
        # Same as FunctionDef?
        body_tree = self.visit(node.body)
        return body_tree

    def visit_Return(self, node):
        return self.visit(node.value)

    def visit_UnaryOp(self, node):
        supported = {
            ast.Not: (lambda opr: Not(opr))
        }
        op_func = supported[type(node.op)]
        opr = self.visit(node.operand)
        return op_func(opr)

    def visit_Subscript(self, node):
        # TODO check value is the variable?
        return VariableUse(node.slice.value.n)

    def visit_BoolOp(self, node):
        supported = {
            ast.And: (lambda left, right: And(left, right)),
            ast.Or: (lambda left, right: Or(left, right))
        }
        op_func = supported[type(node.op)]
        ltree = self.visit(node.values[0])
        rtree = self.visit(node.values[1])
        return op_func(ltree, rtree)
