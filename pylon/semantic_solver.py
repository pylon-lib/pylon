import torch
import torch.nn.functional as f
import torch._dynamo as dynamo
from typing import Dict, List, Tuple
import sys, os
from .utils.tensorize_circuit import tensorize_circuit
from .solver import Solver

torch.set_float32_matmul_precision('high')
dynamo.config.cache_size_limit=10000
sys.setrecursionlimit(20000)

@torch.compile(fullgraph=True, mode='reduce-overhead', dynamic=True)
def levelwiseSL(levels: List[torch.Tensor], idx2primesub: torch.Tensor, data: torch.Tensor):
    for i, level in enumerate(levels):
        data[level] = data[idx2primesub[level]].sum(-2).logsumexp(-2)
    return data[levels[-1]]

class SemanticSolver(Solver):
    def __init__(self, vtree_file, circuit_file):
        circuit = tensorize_circuit(vtree_file, circuit_file)
        self.ID = circuit['ID']
        self.levels = circuit['levels']
        self.idx2primesub = circuit['idx2primesub']
        self.true_indices = circuit['true_indices']
        self.literal_indices = circuit['literal_indices']
        self.literal_mask = circuit['literal_mask']

    def loss(self, logits):
        lit_weights = logits#f.logsigmoid(logits)

        batch_size = lit_weights.shape[-1]
        data = torch.empty(self.ID+1, batch_size, device='cuda')
        data[self.true_indices] = 0
        data[self.ID] = -1000
        data[self.literal_indices] = lit_weights[self.literal_mask[0], self.literal_mask[1]]
        res_sl = levelwiseSL(self.levels, self.idx2primesub, data)

        return (-res_sl).mean()
