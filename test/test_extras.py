import unittest

import torch
from torch.autograd import Variable

import torch_extras

setattr(torch, 'expand_dims', torch_extras.expand_dims)
setattr(torch, 'select_item', torch_extras.select_item)


class PytorchExtrasTestCase(unittest.TestCase):

    def test_expand_dims_variable(self):
        var = Variable(torch.range(0, 9).view(-1, 2))
        ret = torch.expand_dims(var, 0)
        expected = var.view(1, -1, 2)
        assert ret.size() == expected.size()

    def test_expand_dims_tensor(self):
        var = torch.range(0, 9).view(-1, 2)
        ret = torch.expand_dims(var, 0)
        expected = var.view(1, -1, 2)
        assert ret.size() == expected.size()

    def test_select_item_variable(self):
        var = Variable(torch.range(0, 9).view(-1, 2))
        index = torch.Tensor([0, 0, 0, 1, 1]).long()
        ret = torch.select_item(var, index)
        expected = torch.Tensor([0, 2, 4, 7, 9])
        assert all(torch.eq(ret.data, expected))

    def test_select_item_tensor(self):
        var = torch.range(0, 9).view(-1, 2)
        index = torch.Tensor([0, 0, 0, 1, 1]).long()
        ret = torch.select_item(var, index)
        expected = torch.Tensor([0, 2, 4, 7, 9])
        assert all(torch.eq(ret, expected))


if __name__ == '__main__':
    unittest.main()
