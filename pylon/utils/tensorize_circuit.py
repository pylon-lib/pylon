import torch
import sys, os

# Loading the constraint using pypsdd
sys.path.append(os.path.join(sys.path[0], '/space/ahmedk/arsl/autoregressive-semantic-loss/detox/pypsdd'))
from pypsdd import Vtree, SddManager, PSddManager, io

def tensorize_circuit(vtree_file, circuit_file):

    # Load constraint
    vtree = Vtree.read(vtree_file)
    manager = SddManager(vtree)
    alpha = io.sdd_read(circuit_file, manager)
    pmanager = PSddManager(vtree)
    beta = pmanager.copy_and_normalize_sdd(alpha, vtree)

    # Iterate over the circuit's nodes
    nodes = [node for node in beta.positive_iter()]

    # Assign sequential ids to all the nodes in the circuit
    id = 1
    for e in nodes:
        e.id = id
        id += 1

    # Construct levels using topological sort
    import graphlib
    topo = graphlib.TopologicalSorter()
    for node in beta.positive_iter():
        if node.is_decomposition():
            for p, s in node.positive_elements:
                if p.is_decomposition(): topo.add(node, p)
                if s.is_decomposition(): topo.add(node, s)

    topo.prepare()
    levels_nodes = []
    while topo.is_active():
        nodes_level = topo.get_ready()
        topo.done(*nodes_level)
        levels_nodes.append(list(nodes_level))


    # Determine amount of padding needed
    max_elements = 0
    for node in nodes:
        if node.is_decomposition():
            max_elements = max(max_elements, len(node.positive_elements))

    levels = []
    for level in levels_nodes:
        levels.append(torch.tensor([l.id for l in level], dtype=torch.long, device='cuda'))

    true_indices = torch.LongTensor([node.id for node in nodes if node.is_true()]).cuda().unique()

    literal_indices = torch.LongTensor([[node.id, node.literal] for node in nodes if node.is_literal()]).cuda()
    literal_indices, literal_mask = literal_indices.unbind(-1)

    literals = literal_mask.detach().clone()
    literal_mask = literal_mask.abs() - 1, (literal_mask > 0).long()

    idx2primesub = torch.zeros((id, max_elements, 2), dtype=torch.int)
    for node in nodes:
        if node.is_decomposition():
            tmp = torch.LongTensor([[p.id, s.id] for p, s in node.positive_elements])
            idx2primesub[node.id] = torch.nn.functional.pad(tmp, (0,0,0, max_elements - len(tmp)), value=id)
    idx2primesub = idx2primesub.cuda()

    ID = id

    return {'levels': levels, 'true_indices': true_indices, 'literal_indices': literal_indices, 
            'literals': literals, 'literal_mask': literal_mask, 'idx2primesub': idx2primesub, 'ID': ID}
