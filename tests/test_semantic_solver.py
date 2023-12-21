from pylon.constraint import constraint
from pylon.semantic_solver import *

from pysdd.sdd import SddManager
def create_constraint():
    # Specify the number of variable in `var_count`
    mgr = SddManager(var_count=3, auto_gc_and_minimize=False)

    # The ith variable can alternatively be accessed as mgr.vars[i]
    X1, X2, X3 = mgr.vars

    # A constraint specifying exactly 2 out of 3
    alpha = (X1 & ((X2 & -X3) | (-X2 & X3))) | (-X1 & (X2 & X3))

    # Saving circuit & vtree to disk
    alpha.save(str.encode('alpha.sdd'))
    alpha.vtree().save(str.encode('alpha.vtree'))

def test_semantic_solver():

    create_constraint()
    constraint_loss = constraint(None, SemanticSolver('alpha.vtree', 'alpha.sdd'))

    # A tensor of shape (num_literals, 2, batch_size), where the second
    # dimensions corresponds to [log_p(-Xi), log_p(Xi)]
    lit_weights = torch.tensor([[0.6, 0.4], [0.7, 0.3], [0.1, 0.9]], 
            device=torch.cuda.current_device()).log()
    lit_weights = lit_weights.unsqueeze(-1)

    print(constraint_loss(lit_weights))

test_semantic_solver()
