#!/usr/bin/env python

from os import path
import locale # for printing numbers with commas
locale.setlocale(locale.LC_ALL, "en_US.UTF8")
from pypsdd import Timer,Vtree,SddManager,PSddManager,SddNode,PSddNode,Inst
from pypsdd import Prior,DirichletPrior,UniformSmoothing
from pypsdd import io

def fmt(number):
    return locale.format("%d",number,grouping=True)

def run_test(vtree_filename,sdd_filename,seed=0,enum_models=0):

    # READ SDD
    with Timer("reading vtree and sdd"):
        vtree = Vtree.read(vtree_filename)
        manager = SddManager(vtree)
        alpha = io.sdd_read(sdd_filename,manager)

    # CONVERT TO PSDD
    with Timer("converting to two psdds"):
        pmanager1 = PSddManager(vtree)
        pmanager2 = PSddManager(vtree)
        beta  = pmanager1.copy_and_normalize_sdd(alpha,vtree)
        gamma = pmanager2.copy_and_normalize_sdd(alpha,vtree)
        #prior = DirichletPrior(2.0)
        prior = UniformSmoothing(1.0)
        prior.initialize_psdd(beta)
        Prior.random_parameters(gamma,seed=(seed+1))

    # PRINT SOME STATS
    print("================================")
    print(" sdd model count: %s" % fmt(alpha.model_count(vtree)))
    print("       sdd count: %s" % fmt(alpha.count()))
    print("        sdd size: %s" % fmt(alpha.size()))
    print("================================")
    print("psdd model count: %s" % fmt(beta.model_count()))
    print("      psdd count: %s" % fmt(beta.count()))
    print("       psdd size: %s" % fmt(beta.size()))
    print("================================")
    print("     theta count: %s" % fmt(beta.theta_count()))
    print("      zero count: %s" % fmt(beta.zero_count()))
    print("      true count: %s" % fmt(beta.true_count()))

    if beta.vtree.var_count <= PSddNode._brute_force_limit:
        print("=== beta ===")
        print(beta.as_table())
        print("=== gamma ===")
        print(gamma.as_table())
        print("=== end ===")

        print("brute force:")
        print("kl(beta,gamma)  = %.8g" % beta.kl_psdd_brute_force(gamma))
        print("kl(gamma,beta)  = %.8g" % gamma.kl_psdd_brute_force(beta))
        print("kl(beta,beta)   = %.8g" % beta.kl_psdd_brute_force(beta))
        print("kl(gamma,gamma) = %.8g" % gamma.kl_psdd_brute_force(gamma))
    print("compute:")
    print("kl(beta,gamma)  = %.8g" % beta.kl_psdd(gamma))
    print("kl(gamma,beta)  = %.8g" % gamma.kl_psdd(beta))
    print("kl(beta,beta)   = %.8g" % beta.kl_psdd(beta))
    print("kl(gamma,gamma) = %.8g" % gamma.kl_psdd(gamma))

    print("compute:")
    print("kl(beta,gamma)  = %.8g" % beta.kl_psdd_alt(gamma))
    print("kl(gamma,beta)  = %.8g" % gamma.kl_psdd_alt(beta))
    print("kl(beta,beta)   = %.8g" % beta.kl_psdd_alt(beta))
    print("kl(gamma,gamma) = %.8g" % gamma.kl_psdd_alt(gamma))

    ess = 2.0
    prior = UniformSmoothing(ess)
    print("log prior (ess=%.8f,mc=%d):" % (ess,beta.model_count()))
    if beta.vtree.var_count <= PSddNode._brute_force_limit:
        print("method 1 = %.8g" % prior.log_prior_brute_force(beta))
    print("method 2 = %.8g" % prior.log_prior(beta))

    return beta,pmanager1

def run_test_basename(basename,enum_models=1000):
    print("######## " + basename)
    dirname = path.join(path.dirname(__file__),'examples')
    vtree_filename = path.join(dirname,basename + '.vtree')
    sdd_filename = path.join(dirname,basename + '.sdd')
    alpha,pmanager = run_test(vtree_filename,sdd_filename,enum_models=enum_models)
    print()
    return alpha,pmanager

if __name__ == '__main__':
    alpha,pmanager = run_test_basename('ranking-3')
    alpha,pmanager = run_test_basename('example')
    alpha,pmanager = run_test_basename('true')
    alpha,pmanager = run_test_basename('literal')
    alpha,pmanager = run_test_basename('false')
    alpha,pmanager = run_test_basename('xor-16')
    run_test_basename('xor-32')
    run_test_basename('alarm')
    run_test_basename('c432')
