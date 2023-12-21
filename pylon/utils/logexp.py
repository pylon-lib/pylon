@torch.jit.script
def log1mexp(x):
    lt = (x < 0.6931471805599453094).logical_and(x > 0)
    gt = x >= 0.6931471805599453094
    res = torch.empty_like(x)
    res[lt] = torch.log(-torch.expm1(-x[lt]))
    res[gt] = torch.log1p(-torch.exp(-x[gt]))
    res = res.masked_fill_(x == 0, -float('inf'))
    return res
