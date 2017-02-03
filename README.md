# pytorch-extras

`pip install pytorch-extras`

## Usage

### [expand_along](#expand_along)

`expand_along(var, mask)` - Useful for selecting a dynamic amount of items from different indexes using a byte mask.

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
