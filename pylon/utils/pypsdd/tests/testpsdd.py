#!/usr/bin/env python

from os import path
import locale # for printing numbers with commas
locale.setlocale(locale.LC_ALL, "en_US.UTF8")
from pypsdd import Vtree,SddManager,PSddManager,SddNode
from pypsdd import Timer,DataSet,Inst,InstMap
from pypsdd import Prior,DirichletPrior,UniformSmoothing
from pypsdd import io

def fmt(number):
    return locale.format("%d",number,grouping=True)

def run_test(vtree_filename,sdd_filename,N=1024,seed=0,
             print_models=True,test_learning=True):

    # READ SDD
    with Timer("reading vtree and sdd"):
        vtree = Vtree.read(vtree_filename)
        manager = SddManager(vtree)
        alpha = io.sdd_read(sdd_filename,manager)

    # CONVERT TO PSDD
    with Timer("converting to psdd"):
        pmanager = PSddManager(vtree)
        beta = pmanager.copy_and_normalize_sdd(alpha,vtree)

    if test_learning:
        # SIMULATE DATASETS
        with Timer("simulating datasets"):
            #prior = DirichletPrior(2.0)
            prior = UniformSmoothing(1024.0)
            prior.initialize_psdd(beta)
            training = DataSet.simulate(beta,N,seed=seed)
            testing  = DataSet.simulate(beta,N,seed=(seed+1))

        # LEARN A PSDD
        with Timer("learning complete data"):
            beta.learn(training,prior)

        with Timer("evaluate log likelihood"):
            train_ll = beta.log_likelihood(training)/training.N
            test_ll = beta.log_likelihood(testing)/testing.N

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

    if test_learning:
        print("================================")
        print("   training size: %d" % training.N)
        print("    testing size: %d" % testing.N)
        print(" unique training: %d" % len(training))
        print("  unique testing: %d" % len(testing))
        print("================================")
        print("     training ll: %.8f" % train_ll)
        print("      testing ll: %.8f" % test_ll)

        print("================================")
        print(training)

    value = beta.value()
    print("================================")
    print("      p(T) value: %.8f" % beta.value())

    e_inst = Inst.from_literal(1,pmanager.var_count)
    pval = beta.value(evidence=e_inst)
    e_inst = Inst.from_literal(-1,pmanager.var_count)
    nval = beta.value(evidence=e_inst)
    print("p(a)+p(~a) value: %.8f" % (pval+nval))
    print("      p(a) value: %.8f" % pval)
    print("     p(~a) value: %.8f" % nval)
    if value:
        print("     probability: %.8f" % beta.probability(evidence=e_inst))

    var_marginals = beta.marginals()
    value = var_marginals[0]
    check = True
    for var in range(1,pmanager.var_count+1):
        e_inst = Inst.from_literal(1,pmanager.var_count)
        pval = beta.value(evidence=e_inst)
        e_inst = Inst.from_literal(-1,pmanager.var_count)
        nval = beta.value(evidence=e_inst)
        if abs(pval+nval - value) > 1e-8: check = False
    assert check
    print(" marginals check: %s" % ("ok" if check else "NOT OK"))

    inst = InstMap()
    inst[1] = 1
    inst[pmanager.var_count] = 0
    var_marginals = beta.marginals(evidence=inst)
    value = var_marginals[0]
    check = True
    for var in range(2,pmanager.var_count):
        inst[var] = 1
        pval = beta.value(evidence=inst)
        inst[var] = 0
        nval = beta.value(evidence=inst)
        del inst[var]
        if abs(pval+nval - value) > 1e-8: check = False
    assert check
    print(" marginals check: %s" % ("ok" if check else "NOT OK"))

    return beta,pmanager

def run_test_basename(basename,test_learning=True):
    print("######## " + basename)
    dirname = path.join(path.dirname(__file__),'examples')
    vtree_filename = path.join(dirname,basename + '.vtree')
    sdd_filename = path.join(dirname,basename + '.sdd')
    alpha,pmanager = run_test(vtree_filename,sdd_filename,\
                              test_learning=test_learning)
    print()
    return alpha,pmanager

if __name__ == '__main__':
    alpha,pmanager = run_test_basename('ranking-3')
    alpha,pmanager = run_test_basename('example')
    run_test_basename('true')
    run_test_basename('literal')
    run_test_basename('false',test_learning=False)
    run_test_basename('xor-16')
    run_test_basename('xor-32')
    run_test_basename('alarm')
    run_test_basename('c432')
