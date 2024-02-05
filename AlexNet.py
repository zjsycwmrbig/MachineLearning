import torch
import torch.nn as nn
from d2l import torch as d2l

net = nn.Sequential(
    # 这里使用了更大的11 x 11窗口来捕获对象。同时，步幅为4，以减少输出的高度和宽度。
    nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1), nn.ReLU(),
    # 减少卷积窗口，使用填充为2来使得输入与输出的高和宽一致，且增大输出通道数。
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    # 连续三个卷积层，且使用更小的卷积窗口。除了最后的卷积层外，进一步增大了输出通道数。
    nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),nn.Flatten(),


    # 这里，全连接层的输出数量是LeNet中的好几倍。使用dropout层来减轻过拟合。
    nn.Linear(6400, 4096), nn.ReLU(), nn.Dropout(p=0.5),
    nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(p=0.5),
    # 最后是输出层。由于这里使用Fashion-MNIST，所以用类别数为10，而非论文中的1000。
    nn.Linear(4096, 10)
)

batch_size = 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size, resize=224)
# 这里的学习率设置的比较小，因为AlexNet使用了较大的minibatch 网络更复杂
lr, num_epochs = 0.01, 10


def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)


if __name__ == '__main__':
    # 这里使用了Xavier随机初始化参数。这个初始化方法的提出是为了更有效地利用隐藏层的激活函数。
    loss = nn.CrossEntropyLoss()
    trainer = torch.optim.SGD(net.parameters(), lr=lr)
    net.apply(init_weights)
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
