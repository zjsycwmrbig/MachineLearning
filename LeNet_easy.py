import torch
from torch import nn
from d2l import torch as d2l

# 复现90年代的经典网络LeNet 这里的Sigmoid 激活函数在现在看来有点过时了
net_sigmoid = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5, padding=2),
    nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5),
    nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(16 * 5 * 5, 120),
    nn.Sigmoid(),
    nn.Linear(120, 84),
    nn.Sigmoid(),
    nn.Linear(84, 10)
)
net_relu = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5, padding=2),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(16 * 5 * 5, 120),
    nn.ReLU(),
    nn.Linear(120, 84),
    nn.ReLU(),
    nn.Linear(84, 10)
)

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)


def train(net, train_iter, test_iter, num_epochs, lr):
    # 初始化模型参数
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)

    net.apply(init_weights)
    loss = nn.CrossEntropyLoss()
    trainer = torch.optim.SGD(net.parameters(), lr=lr)
    for epoch in range(num_epochs):
        for X, y in train_iter:
            l = loss(net(X), y)
            trainer.zero_grad()
            l.backward()
            trainer.step()
        with torch.no_grad():
            acc_sum, n = 0.0, 0
            for X, y in test_iter:
                acc = (net(X).argmax(axis=1) == y).sum()
                acc_sum += acc
                n += y.shape[0]
            print(f'epoch {epoch + 1}, acc {acc_sum / n:.3f}')


if __name__ == '__main__':
    lr, num_epochs = 0.3, 10
    # 0.815
    # train(net_sigmoid, train_iter, test_iter, num_epochs, lr)
    # 学习率降低到降低到 0.3 0.884
    train(net_relu, train_iter, test_iter, num_epochs, lr)