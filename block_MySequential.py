from torch import nn


class MySequential(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        for idx, layer in enumerate(kwargs):
            self._modules[str(idx)] = layer

    def forward(self, X):
        # 重点是保证是顺序的
        for block in self._modules.values():
            X = block(X)
        return X

