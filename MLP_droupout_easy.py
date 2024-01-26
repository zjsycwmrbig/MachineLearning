import torch
from torch import nn
from d2l import torch as d2l

net = nn.Sequential(nn.Flatten(),
                    nn.Linear(784, 256),
                    nn.ReLU(),
                    # 在第一个全连接层之后添加一个dropout层
                    nn.Dropout(0.5),
                    nn.Linear(256, 256),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    # 在第二个全连接层之后添加一个dropout层
                    nn.Linear(256, 10))

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights)

trainer = torch.optim.SGD(net.parameters(), lr=0.5)
loss = nn.CrossEntropyLoss(reduction='none')
num_epochs = 10
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

if __name__ == '__main__':
    for epoch in range(num_epochs):
        for X, y in train_iter:
            l = loss(net(X), y)
            trainer.zero_grad()
            l.mean().backward()
            trainer.step()
        with torch.no_grad():
            total_acc = 0
            for X, y in test_iter:
                acc = (net(X).argmax(axis=1) == y).sum()
                total_acc += acc
            print(f'epoch {epoch + 1}, acc {total_acc / len(test_iter.dataset):.3f}')