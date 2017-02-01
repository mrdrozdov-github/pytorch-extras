# pytorch-extras

`pip install pytorch-extras`

## Usage

`select_item(var, index)` - Is similar to `[var[row,col] for row, col in enumerate(index)]`.

    import torch
    import torch_extras
    setattr(torch, 'select_item', torch_extras.select_item)

    var = torch.range(0, 9).view(-1, 2)
    index = torch.LongTensor([0, 0, 0, 1, 1])
    ret = torch.select_item(var, index)
    # [0, 2, 4, 7, 9]
