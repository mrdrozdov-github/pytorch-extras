import six
import torch
from torch.autograd import Variable


def expand_along(var, mask):
    """ Useful for selecting a dynamic amount of items from different
        indexes using a byte mask.

        ```
        import torch
        import torch_extras
        setattr(torch, 'expand_along', torch_extras.expand_along)

        var = torch.Tensor([1, 0, 2])
        mask = torch.ByteTensor([[True, True], [False, True], [False, False]])
        torch.expand_along(var, mask)
        # (1, 1, 0)
        ```
    """
    indexes = torch.range(0, var.size(0) - 1).view(-1, 1).repeat(1, mask.size(1))
    _mask = indexes[mask].long()
    if isinstance(var, Variable):
        _mask = Variable(_mask, volatile=var.volatile)
    return torch.index_select(var, 0, _mask)


def expand_dims(var, dim=0):
    """ Is similar to [numpy.expand_dims](https://docs.scipy.org/doc/numpy/reference/generated/numpy.expand_dims.html).

        import torch
        import torch_extras
        setattr(torch, 'expand_dims', torch_extras.expand_dims)

        var = torch.range(0, 9).view(-1, 2)
        torch.expand_dims(var, 0).size()
        # (1, 5, 2)
    """
    sizes = list(var.size())
    sizes.insert(dim, 1)
    return var.view(*sizes)


def select_item(var, index):
    """ Is similar to `[var[row,col] for row, col in enumerate(index)]`.

        ```
        import torch
        import torch_extras
        setattr(torch, 'select_item', torch_extras.select_item)

        var = torch.range(0, 9).view(-1, 2)
        index = torch.LongTensor([0, 0, 0, 1, 1])
        torch.select_item(var, index)
        # [0, 2, 4, 7, 9]
        ```
    """
    index_mask = index.view(-1, 1).repeat(1, var.size(1))
    mask = torch.range(0, var.size(1) - 1).long()
    mask = mask.repeat(var.size(0), 1)
    mask = mask.eq(index_mask)
    if isinstance(var, Variable):
        mask = Variable(mask, volatile=var.volatile)
    return torch.masked_select(var, mask)


def cast(var, type):
    """ Cast a Tensor to the given type.

        ```
        import torch
        import torch_extras
        setattr(torch, 'cast', torch_extras.cast)

        input = torch.FloatTensor(1)
        target_type = type(torch.LongTensor(1))
        type(torch.cast(input, target_type))
        # <class 'torch.LongTensor'>
        ```
    """
    if type == torch.ByteTensor:
        return var.byte()
    elif type == torch.CharTensor:
        return var.char()
    elif type == torch.DoubleTensor:
        return var.double()
    elif type == torch.FloatTensor:
        return var.float()
    elif type == torch.IntTensor:
        return var.int()
    elif type == torch.LongTensor:
        return var.long()
    elif type == torch.ShortTensor:
        return var.short()
    else:
        raise ValueError("Not a Tensor type.")


def one_hot(size, index):
    """ Creates a matrix of one hot vectors.

        ```
        import torch
        import torch_extras
        setattr(torch, 'one_hot', torch_extras.one_hot)

        size = (3, 3)
        index = torch.LongTensor([2, 0, 1]).view(-1, 1)
        torch.one_hot(size, index)
        # [[0, 0, 1], [1, 0, 0], [0, 1, 0]]
        ```
    """
    mask = torch.LongTensor(*size).fill_(0)
    ones = 1
    if isinstance(index, Variable):
        ones = Variable(torch.LongTensor(index.size()).fill_(1))
        mask = Variable(mask, volatile=index.volatile)
    ret = mask.scatter_(1, index, ones)
    return ret


def nll(log_prob, label):
    """ Is similar to [`nll_loss`](http://pytorch.org/docs/nn.html?highlight=nll#torch.nn.functional.nll_loss) except does not return an aggregate.

        ```
        import torch
        from torch.autograd import Variable
        import torch.nn.functional as F
        import torch_extras
        setattr(torch, 'nll', torch_extras.nll)

        input = Variable(torch.FloatTensor([[0.5, 0.2, 0.3], [0.1, 0.8, 0.1]]))
        target = Variable(torch.LongTensor([1, 2]).view(-1, 1))
        output = torch.nll(torch.log(input), target)
        output.size()
        # (2, 1)
        ```
    """
    if isinstance(log_prob, Variable):
        _type = type(log_prob.data)
    else:
        _type = type(log_prob)

    mask = one_hot(log_prob.size(), label)
    mask = cast(mask, _type)
    return -1 * (log_prob * mask).sum(1)
