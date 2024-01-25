import torch
from torch import nn
from d2l import torch as d2l
# 两层神经网络 这里隐藏层的神经元个数为256 为什么是256呢？？？

net = nn.Sequential(nn.Flatten(), nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10))


# init_weights(net)  # 初始化模型参数
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)


net.apply(init_weights)

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
loss = nn.CrossEntropyLoss(reduction='none')
trainer = torch.optim.SGD(net.parameters(), lr=0.1)
num_epochs = 10
if __name__ == '__main__':
    for epoch in range(num_epochs):
        for X, y in train_iter:
            l = loss(net(X), y)
            l = torch.mean(l)  # 计算平均损失
            trainer.zero_grad()
            l.backward()
            trainer.step()
        with torch.no_grad():
            total_acc = 0
            for X, y in test_iter:
                acc = (net(X).argmax(axis=1) == y).sum()
                total_acc += acc
            print(f'epoch {epoch + 1}, acc {total_acc / len(test_iter.dataset):.3f}')