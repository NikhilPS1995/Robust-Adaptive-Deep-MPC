import torch
import numpy as np

def torchify(*args, cls=torch.FloatTensor, device=None):
    out = []
    for x in args:
        if type(x) is not torch.Tensor and type(x) is not np.ndarray:
            x = np.array(x)
        if type(x) is not torch.Tensor:
            x = cls(x)
        if device is not None:
            x = x.to(device)
        out.append(x)
    if len(out) == 1:
        return out[0]
    else:
        return tuple(out)