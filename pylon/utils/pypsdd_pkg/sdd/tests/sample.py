#!/usr/bin/env python

import glob
from pylab import *

from pypsdd import Vtree,SddManager,PSddManager,io
from pypsdd import DataSet,Prior,DirichletPrior,UniformSmoothing

k = 50 # number of training sets
Ns = list(range(8,13)) # dataset sizes
vtree_filename = "pypsdd/tests/examples/example.vtree"
sdd_filename = "pypsdd/tests/examples/example.sdd"

print("reading vtree and sdd ...")
vtree = Vtree.read(vtree_filename)
manager = SddManager(vtree)
alpha = io.sdd_read(sdd_filename,manager)

print("converting to two psdds ...")
pmanager1 = PSddManager(vtree)
pmanager2 = PSddManager(vtree)
beta  = pmanager1.copy_and_normalize_sdd(alpha,vtree)
gamma = pmanager2.copy_and_normalize_sdd(alpha,vtree)
Prior.random_parameters(beta) # randomly parameterize beta

print("simulating datasets from beta ...")
# for each N, simulate a set of k datasets
train_sets = [ [DataSet.simulate(beta,2**N) for i in range(k)] for N in Ns ]

print("running learning experiments ...")
results = []
prior = DirichletPrior(2.0)
for batch in train_sets: # for each N in Ns
    batch_results = []
    for train in batch: # for each training set in batch
        gamma.learn(train,prior)
        kl = beta.kl_psdd(gamma)
        batch_results.append(kl)
    results.append(batch_results)


# settings for plotting
fontsize = 22
rcParams.update({'xtick.labelsize'   : fontsize-4,
                 'ytick.labelsize'   : fontsize-4,
                 'axes.labelsize'    : fontsize,
                 'axes.titlesize'    : fontsize,
                 'legend.fontsize'   : fontsize-6,
                 'figure.autolayout' : True,
                 'pdf.fonttype'      : 42})

print("plotting ...")
X = Ns
Y = [ average(batch) for batch in results ]
plot(X,Y,'b-',label='psdd',linewidth=3)

title('Recoverability of PSDDs')
xlabel('dataset size (2^x)')
ylabel('KL(P,Q)')
axes().set_yscale('log')
legend(loc='best')
savefig('plot.png')
savefig('plot.pdf')
