# pytorch-extras

`pip install pytorch-extras`

## Usage

### [expand_along](#expand_along)

`expand_along(var, mask)` - Useful for selecting a dynamic amount of items from different indexes using a byte mask. This is a bit like [numpy.repeat](https://docs.scipy.org/doc/numpy/reference/generated/numpy.repeat.html).

    import torch
    import torch_extras
    setattr(torch, 'expand_along', torch_extras.expand_along)

    var = torch.Tensor([1, 0, 2])
    mask = torch.ByteTensor([[True, True], [False, True], [False, False]])
    torch.expand_along(var, mask)
    # (1, 1, 0)


### [expand_dims](#expand_dims)

`expand_dims(var, dim)` - Is similar to [numpy.expand_dims](https://docs.scipy.org/doc/numpy/reference/generated/numpy.expand_dims.html).

    import torch
    import torch_extras
    setattr(torch, 'expand_dims', torch_extras.expand_dims)

    var = torch.range(0, 9).view(-1, 2)
    torch.expand_dims(var, 0).size()
    # (1, 5, 2)
    
Note: Have recently found out about [torch.unsqeeze](http://pytorch.org/docs/tensors.html?highlight=unsqueeze#torch.Tensor.unsqueeze), which has the same API and is probably a more effective method for expanding dimensions.


### [select_item](#select_item)

`select_item(var, index)` - Is similar to `[var[row,col] for row, col in enumerate(index)]`.

    import torch
    import torch_extras
    setattr(torch, 'select_item', torch_extras.select_item)

    var = torch.range(0, 9).view(-1, 2)
    index = torch.LongTensor([0, 0, 0, 1, 1])
    torch.select_item(var, index)
    # [0, 2, 4, 7, 9]


### [cast](#cast)

`cast(var, type)` - Cast a Tensor to the given type.

    import torch
    import torch_extras
    setattr(torch, 'cast', torch_extras.cast)

    input = torch.FloatTensor(1)
    target_type = type(torch.LongTensor(1))
    type(torch.cast(input, target_type))
    # <class 'torch.LongTensor'>


### [one_hot](#one_hot)


`one_hot(size, index)` - Creates a matrix of one hot vectors.

    import torch
    import torch_extras
    setattr(torch, 'one_hot', torch_extras.one_hot)

    size = (3, 3)
    index = torch.LongTensor([2, 0, 1]).view(-1, 1)
    torch.one_hot(size, index)
    # [[0, 0, 1], [1, 0, 0], [0, 1, 0]]


### [nll](#nll)

`nll(log_prob, label)` - Is similar to [`nll_loss`](http://pytorch.org/docs/nn.html?highlight=nll#torch.nn.functional.nll_loss) except does not return an aggregate.

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
