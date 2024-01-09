import torch
import torch.fx
from pysdd.sdd import SddManager, Vtree
import inspect
import operator

def find_terminal_node(graph):
    for node in graph.nodes:
        if (not node.users) and (node.op == 'output'):  # Check if the node has no users
            return node

def capture_constraint(m: torch.nn.Module):
    graph : torch.fx.Graph = torch.fx.symbolic_trace(m).graph
    input_names = list(inspect.signature(m).parameters.keys())
    root_node = find_terminal_node(graph)

    mgr = SddManager(var_count=len(input_names), auto_gc_and_minimize=True)

    return depth_first_search(graph, root_node, input_names, mgr)

def depth_first_search(graph, node, input_names, mgr):

  if node.op == 'call_function':

    if node.target == torch.logical_and or node.target == operator.__and__:
      alpha = depth_first_search(graph, node.args[0], input_names, mgr)
      beta = depth_first_search(graph, node.args[1], input_names, mgr)
      alpha.ref(); beta.ref()

      gamma = alpha & beta
      gamma.ref(); alpha.deref(); beta.deref()

      return gamma

    elif node.target == torch.logical_or or node.target == operator.__or__:
      alpha = depth_first_search(graph, node.args[0], input_names, mgr)
      beta = depth_first_search(graph, node.args[1], input_names, mgr)
      alpha.ref(); beta.ref()

      gamma = alpha | beta
      gamma.ref(); alpha.deref(); beta.deref()

      return gamma

    elif node.target == torch.logical_not or node.target == operator.__invert__:
      alpha =  depth_first_search(graph, node.args[0], input_names, mgr)
      alpha.ref()

      gamma = -depth_first_search(graph, node.args[0], input_names, mgr)
      gamma.ref(); alpha.deref()

      return gamma

  elif node.op == 'output':
      return depth_first_search(graph, node.args[0], input_names, mgr)

  elif node.op == 'placeholder':
      index = input_names.index(node.name) + 1

      return mgr.vars[index]
