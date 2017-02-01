# pytorch-extras

`pip install pytorch-extras`

## Usage

### [expand_dims](#expand_dims)

`expand_dims(var, dim)` - Is similar to [numpy.expand_dims](https://docs.scipy.org/doc/numpy/reference/generated/numpy.expand_dims.html).

    import torch
    import torch_extras
    setattr(torch, 'expand_dims', torch_extras.expand_dims)

    var = torch.range(0, 9).view(-1, 2)
    torch.expand_dims(var, 0).size()
    # (1, 5, 2)


### [select_item](#select_item)

`select_item(var, index)` - Is similar to `[var[row,col] for row, col in enumerate(index)]`.

    import torch
    import torch_extras
    setattr(torch, 'select_item', torch_extras.select_item)

    var = torch.range(0, 9).view(-1, 2)
    index = torch.LongTensor([0, 0, 0, 1, 1])
    torch.select_item(var, index)
    # [0, 2, 4, 7, 9]
