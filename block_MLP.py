import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(784, 256)
        self.act = nn.ReLU()
        self.output = nn.Linear(256, 10)

    # 这里的forward函数是必须的 表明怎么进行前向传播
    def forward(self, x):
        a = self.act(self.hidden(x))
        return self.output(a)


X = torch.rand(2, 784)
net = MLP()
print(net(X))



