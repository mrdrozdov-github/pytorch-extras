import six
import torch
from torch.autograd import Variable


def expand_along(var, mask):
    indexes = torch.range(0, var.size(0) - 1).view(-1, 1).repeat(1, mask.size(1))
    _mask = indexes[mask].long()
    if isinstance(var, Variable):
        _mask = Variable(_mask, volatile=var.volatile)
    return torch.index_select(var, 0, _mask)


def expand_dims(var, dim=0):
    sizes = list(var.size())
    sizes.insert(dim, 1)
    return var.view(*sizes)


def select_item(var, index):
    index_mask = index.view(-1, 1).repeat(1, var.size(1))
    mask = torch.range(0, var.size(1) - 1).long()
    mask = mask.repeat(var.size(0), 1)
    mask = mask.eq(index_mask)
    if isinstance(var, Variable):
        mask = Variable(mask, volatile=var.volatile)
    return torch.masked_select(var, mask)
