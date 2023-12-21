#!/usr/bin/env python

from os import path
import locale # for printing numbers with commas
locale.setlocale(locale.LC_ALL, "en_US.UTF8")
from pypsdd import Timer,Vtree,SddManager,PSddManager,SddNode,Inst
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
    with Timer("converting to psdd"):
        pmanager = PSddManager(vtree)
        beta = pmanager.copy_and_normalize_sdd(alpha,vtree)
        prior = UniformSmoothing(2.0)
        #prior.initialize_psdd(beta)
        Prior.random_parameters(beta,seed=seed)

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

    if beta.vtree.var_count <= 10:
        print(beta.as_table())
    mpe_val,mpe_inst = beta.mpe()
    mpe_val = mpe_val if beta.is_false_sdd else mpe_val/beta.theta_sum
    print("mpe: %s %.8f" % (mpe_inst,mpe_val))

    if enum_models:
        models = []
        with Timer("enumerating %d models" % enum_models):
            for model in beta.enumerate_mpe(pmanager):
                models.append(model)
                if len(models) >= enum_models: break

        for model in models[:10]:
            print(model)
        print("%d models (%d max)" % (len(models),10))

        """
        with Timer("evaluating %d models" % enum_models):
            for model in models:
                if not alpha.is_model(model):
                    print "error: non-model", model
                if not alpha._is_bits_and_data_clear(): # random check
                    print "error: bits or data not clear"
        """
    return beta,pmanager

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
