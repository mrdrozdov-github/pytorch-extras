import torch


def select_item(var, index):
    index_mask = index.view(-1, 1).repeat(1, var.size(1))
    mask = torch.range(0, var.size(1) - 1).long()
    mask = mask.repeat(var.size(0), 1)
    mask = mask.eq(index_mask)
    return var[mask]
