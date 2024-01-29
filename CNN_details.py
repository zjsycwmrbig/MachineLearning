import torch
from torch import nn
from d2l import torch as d2l


def corr2d(X, K):
    """计算二维互相关运算。"""
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i + h, j:j + w] * K).sum()
    return Y


# 二维卷积层
class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(kernel_size))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return corr2d(x, self.weight) + self.bias


def train_kernel():
    # 初始化卷积层
    X = torch.ones((6, 8))
    X[:, 2:6] = 0
    print(X)
    true_kernel = torch.tensor([[1.0, -1.0]])
    Y = corr2d(X, true_kernel)
    print(Y)
    # 通过学习训练我们的K
    conv2d = nn.Conv2d(1, 1, kernel_size=(1, 2), bias=False)
    # 二维卷积层使用四维输入输出格式（批量大小、通道、高度、宽度），这里批量和通道都是1
    X = X.reshape((1, 1, 6, 8))
    Y = Y.reshape((1, 1, 6, 7))

    lr = 3e-2  # 学习率

    for i in range(20):
        Y_hat = conv2d(X)
        l = ((Y_hat - Y) ** 2).sum()
        # 梯度清零
        conv2d.zero_grad()
        l.backward()
        # 更新参数
        conv2d.weight.data[:] -= lr * conv2d.weight.grad
        if (i + 1) % 5 == 0:
            print(f'epoch {i + 1}, loss {l.item():.3f}')


def comp_conv2d(conv2d, X):
    """计算卷积层"""
    # (1, 1)代表批量大小和通道数（“多输入通道和多输出通道”一节将介绍）均为1
    X = X.reshape((1, 1) + X.shape)
    Y = conv2d(X)
    # 排除不关心的前两维：批量和通道
    return Y.reshape(Y.shape[2:])


def try_padding():
    """使用填充 主要的目的是为了保持输入输出的形状一致"""
    # 请注意，这里每边都填充了1行或列，因此总共添加了2行或列
    conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1)
    X = torch.rand(size=(8, 8))
    print(comp_conv2d(conv2d, X).shape)
    # 注意 填充可以是不同的行数和列数 例如 padding=(1,2) 代表上下填充1行 左右填充2列 针对的是内核大小的行列数不同的情况 例如 3*5的内核


def try_step():
    # 这里的3 指的是 3*3的内核
    conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1, stride=2)
    X = torch.rand(size=(8, 8))
    print(comp_conv2d(conv2d, X).shape)


if __name__ == '__main__':
    # train_kernel()
    # try_padding()
    try_step()