import math
import random

class Prior:
    """Abstract parameter prior class.

    Contains some basic functionality."""

    def __init__(self):
        pass

    def initialize_psdd(self,root):
        pass

    def log_prior(self,root):
        pass

    @staticmethod
    def zero_parameters(root):
        """Initialize all parameters to zero."""
        for node in root.positive_iter():
            node.theta = dict( (el,0.0) for el in root.positive_elements )
            node.theta_sum = 0.0

    @staticmethod
    def random_parameter_set(k,psi=1.0):
        """Draws a parameter set from a Dirichlet distribution.
        
        k is the number of states
        psi is the Dirichlet meta-parameter"""
        pr = [ random.gammavariate(psi,1) for i in range(k) ]
        pr_sum = sum(pr)
        return [ p/pr_sum for p in pr ]

    @staticmethod
    def random_parameters(root,psi=1.0,seed=None):
        """For each decision node and true node, draw a parameter set from a
        Dirichlet distribution.

        psi is the Dirichlet meta-parameter."""
        if seed is not None: random.seed(seed)

        for node in root.positive_iter():
            node.theta_sum = 1.0
            if node.is_true():
                node.theta = Prior.random_parameter_set(2,psi=psi)
            elif node.is_literal():
                node.theta = [0.0,0.0]
                node.theta[node.literal > 0] = 1.0
            else: # node.is_decomposition()
                pelements = node.positive_elements
                pr = Prior.random_parameter_set(len(pelements),psi=psi)
                node.theta = dict(list(zip(pelements,pr)))

class DirichletPrior(Prior):
    """Dirichlet prior for PSDDs"""

    def __init__(self,psi=2.0):
        """psi is a Dirichlet parameter.
        psi=1.0 corresponds to zero parameters.
        psi=2.0 corresponds to Laplace (add-one) smoothing."""
        self.psi = psi

    def initialize_psdd(self,root):
        """Initialize the parameters of a PSDD."""
        for node in root.positive_iter():
            self.initialize_node(node)

    def initialize_node(self,node):
        """Initialize the parameters of a PSDD node."""
        count = self.psi-1.0 # pseudo-count
        if node.is_false_sdd:
            pass
        elif node.is_true():
            node.theta = [count,count]
            node.theta_sum = 2.0*count
        elif node.is_literal():
            node.theta = [0.0,0.0]
            node.theta[node.literal > 0] = count
            node.theta_sum = count
        elif node.is_decomposition():
            node.theta = dict( (el,count) for el in node.positive_elements )
            node.theta_sum = count*len(node.positive_elements)

    def log_prior(self,root):
        """Computes log Pr(theta)"""
        if root.is_false_sdd: return float('-inf') # AC: should be 0?

        count = self.psi-1.0
        log_prior = 0.0
        for node in root.positive_iter():
            if node.is_true():
                for theta in node.theta:
                    theta = theta/node.theta_sum
                    log_prior += count*math.log(theta)
            elif node.is_decomposition():
                for element in node.positive_elements:
                    theta = node.theta[element]/node.theta_sum
                    log_prior += count*math.log(theta)

        return log_prior
        
class UniformSmoothing(Prior):
    """Initialize the PSDD weights with a pseudo-count that would be
    equivalent to observing a (prior) dataset over all SDD models,
    with an aggregate size of ESS.

        log Pr(theta) = (ESS/MC) sum_{w \models alpha} log Pr_theta(w)

    where ESS is the equivalent sample size,
    where MC is the model count of alpha.

    Intuitively, we see every model of the SDD in a 'prior'
    dataset, but we pretend the aggregate size of the dataset is
    in total ESS (instead of the model count)."""

    def __init__(self,ess=1.0):
        """ess is the aggregate of the "prior" dataset."""
        self.ess = ess

    def initialize_psdd(self,root):
        """Initialize the parameters of a PSDD."""
        if root.is_false_sdd: return

        # initialize counts
        Prior.zero_parameters(root)
        root.model_count(clear_data=False)

        # propagate equivalent sample size from root to leaves
        root.theta_sum = float(self.ess)
        for node in root.as_positive_list(reverse=True,clear_data=True):
            if node.is_true():
                node.theta = [node.theta_sum/2.0,node.theta_sum/2.0]
            elif node.is_literal():
                node.theta = [0.0,0.0]
                node.theta[node.literal > 0] = node.theta_sum
            else: # node.is_decomposition()
                pelements = node.positive_elements
                mc = float(node.data) # model count
                nc = node.theta_sum   # node count
                counts = [ nc*p.data*s.data/mc for p,s in pelements ]
                node.theta = dict(list(zip(pelements,counts)))
                for (p,s),count in zip(pelements,counts):
                    p.theta_sum += count
                    s.theta_sum += count

    def log_prior(self,root):
        """Computes log Pr(theta)"""
        if root.is_false_sdd: return float('-inf') # AC: should be 0?

        for node in root.as_positive_list(clear_data=False): node.data_sum = 0.0
        mc = root.model_count(clear_data=False)
        root.data_sum = float(mc)

        log_prior = 0.0
        for node in root.as_positive_list(reverse=True,clear_data=True):
            if node.is_true():
                count = 0.5*node.data_sum
                log_prior += count*math.log(node.theta[0]/node.theta_sum)
                log_prior += count*math.log(node.theta[1]/node.theta_sum)
            elif node.is_decomposition():
                for p,s in node.positive_elements:
                    count = node.data_sum*(p.data*s.data/float(node.data))
                    p.data_sum += count
                    s.data_sum += count
                    theta = node.theta[(p,s)]/node.theta_sum
                    log_prior += count*math.log(theta)

        for node in root.as_positive_list(clear_data=False): del node.data_sum
        return (self.ess/float(mc))*log_prior

    def log_prior_brute_force(self,root):
        """Brute-force (enumerative) computation of the 'uniform smoothing'
        parameter prior """
        if root.is_false_sdd: return float('-inf') # AC: should be 0?
        #assert root.vtree.var_count <= PSddNode._brute_force_limit #AC

        mc = root.model_count()
        log_prior = 0.0
        for model in root.models(root.vtree):
            pr = root.pr_model(model)
            log_prior += math.log(pr)
        return (self.ess/float(mc))*log_prior
