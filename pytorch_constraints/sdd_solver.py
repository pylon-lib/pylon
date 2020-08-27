from .tree_node import *


class SddVisitor(TreeNodeVisitor):

    def visit_And(self, node, mgr):
        lv = self.visit(node.left, mgr)
        rv = self.visit(node.right, mgr)
        return lv & rv

    def visit_Or(self, node, mgr):
        lv = self.visit(node.left, mgr)
        rv = self.visit(node.right, mgr)
        return lv | rv

    def visit_Not(self, node, mgr):
        return ~self.visit(node.operand, mgr)

    def visit_IsEq(self, node, mgr):
        return self.visit(node.left, mgr).equiv(self.visit(node.right, mgr))

    def visit_Const(self, node, mgr):
        if node.value == True or node.value == 1:
            return mgr.true()
        elif node.value == False or node.value == 0:
            return mgr.false()
        else:
            raise NotImplementedError

    def visit_FunDef(self, node, mgr):
        return self.visit(node.return_node, mgr)

    def visit_VarUse(self, node, mgr):
        if node.varidx != 0:
            raise NotImplementedError
        else:
            while mgr.var_count() < node.index+1:
                mgr.add_var_after_last()
            return mgr.literal(node.index+1)
