import unittest

import torch
import torch.nn.functional as F
from torch.autograd import Variable

import torch_extras

setattr(torch, 'expand_along', torch_extras.expand_along)
setattr(torch, 'expand_dims', torch_extras.expand_dims)
setattr(torch, 'select_item', torch_extras.select_item)
setattr(torch, 'cast', torch_extras.cast)
setattr(torch, 'one_hot', torch_extras.one_hot)
setattr(torch, 'nll', torch_extras.nll)


class PytorchExtrasTestCase(unittest.TestCase):

    def test_expand_along_variable(self):
        var = Variable(torch.Tensor([1, 0, 2]))
        mask = torch.ByteTensor([[True, True], [False, True], [False, False]])
        ret = torch.expand_along(var, mask)
        expected = torch.Tensor([1, 1, 0])
        assert len(ret) == len(expected)
        assert all(torch.eq(ret.data, expected))

    def test_expand_along_tensor(self):
        var = torch.Tensor([1, 0, 2])
        mask = torch.ByteTensor([[True, True], [False, True], [False, False]])
        ret = torch.expand_along(var, mask)
        expected = torch.Tensor([1, 1, 0])
        assert len(ret) == len(expected)
        assert all(torch.eq(ret, expected))

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
        index = torch.LongTensor([0, 0, 0, 1, 1])
        ret = torch.select_item(var, index)
        expected = torch.Tensor([0, 2, 4, 7, 9])
        assert all(torch.eq(ret.data, expected))

    def test_select_item_tensor(self):
        var = torch.range(0, 9).view(-1, 2)
        index = torch.LongTensor([0, 0, 0, 1, 1])
        ret = torch.select_item(var, index)
        expected = torch.Tensor([0, 2, 4, 7, 9])
        assert all(torch.eq(ret, expected))

    def test_cast(self):
        inputs = [
            torch.ByteTensor(1),
            torch.CharTensor(1),
            torch.DoubleTensor(1),
            torch.FloatTensor(1),
            torch.IntTensor(1),
            torch.LongTensor(1),
            torch.ShortTensor(1),
        ]

        for inp in inputs:
            assert type(inp) == type(torch.cast(inp, type(inp)))

    def test_cast_variable(self):
        inputs = [
            torch.ByteTensor(1),
            torch.CharTensor(1),
            torch.DoubleTensor(1),
            torch.FloatTensor(1),
            torch.IntTensor(1),
            torch.LongTensor(1),
            torch.ShortTensor(1),
        ]

        for inp in inputs:
            assert type(inp) == type(torch.cast(Variable(inp), type(inp)).data)

    def test_one_hot(self):
        size = (3, 3)
        index = torch.LongTensor([2, 0, 1]).view(-1, 1)
        ret = torch.one_hot(size, index)
        expected = torch.LongTensor([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
        assert ret.size() == expected.size()
        assert all(torch.eq(ret.view(-1), expected.view(-1)))

    def test_one_hot_variable(self):
        size = (3, 3)
        index = torch.LongTensor([2, 0, 1]).view(-1, 1)
        ret = torch.one_hot(size, Variable(index))
        expected = torch.LongTensor([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
        assert ret.size() == expected.size()
        assert all(torch.eq(ret.view(-1).data, expected.view(-1)))

    def test_nll(self):
        input = torch.FloatTensor([[0.5, 0.2, 0.3], [0.1, 0.8, 0.1]])
        target = torch.LongTensor([1, 2]).view(-1, 1)
        output = torch.nll(torch.log(input), target)
        assert output.size() == (target.size(0), 1)
        assert all(o == -1 * row[t] for row, o, t in zip(
            torch.log(input), output.view(-1), target.view(-1)))

    def test_nll_variable(self):
        input = Variable(torch.FloatTensor([[0.5, 0.2, 0.3], [0.1, 0.8, 0.1]]))
        target = Variable(torch.LongTensor([1, 2]).view(-1, 1))
        output = torch.nll(torch.log(input), target)
        assert output.size() == (target.size(0), 1)
        assert all(o == -1 * row[t] for row, o, t in zip(
            torch.log(input).data, output.view(-1).data, target.view(-1).data))


if __name__ == '__main__':
    unittest.main()
