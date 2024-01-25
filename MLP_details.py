import torch
from torch import nn
from d2l import torch as d2l

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

numbers_inputs, numbers_outputs, num_hidden = 784, 10, 256

# 两层神经网络
W1 = nn.Parameter(torch.randn(numbers_inputs, num_hidden, requires_grad=True) * 0.01)  # 0.01 权重更小 更容易收敛
b1 = nn.Parameter(torch.zeros(num_hidden, requires_grad=True))
W2 = nn.Parameter(torch.randn(num_hidden, numbers_outputs, requires_grad=True) * 0.01)
b2 = nn.Parameter(torch.zeros(numbers_outputs, requires_grad=True))


def relu(X):
    a = torch.zeros_like(X)
    return torch.max(X, a)


def net(X):
    X = X.reshape((-1, numbers_inputs))
    H = relu(X @ W1 + b1)
    return H @ W2 + b2


loss = nn.CrossEntropyLoss(reduction='none')


if __name__ == '__main__':
    num_epochs, lr = 10, 0.03
    updater = torch.optim.SGD([W1, W2, b1, b2], lr=lr)
    for epoch in range(num_epochs):
        for X, y in train_iter:
            l = loss(net(X), y)
            l = torch.mean(l)
            updater.zero_grad()
            l.backward()
            updater.step()
        with torch.no_grad():
            acc_sum, n = 0.0, 0
            for X, y in test_iter:
                acc = (net(X).argmax(axis=1) == y).sum()
                acc_sum += acc
            print(f'epoch {epoch + 1}, acc {acc_sum / len(test_iter.dataset):.3f}')
