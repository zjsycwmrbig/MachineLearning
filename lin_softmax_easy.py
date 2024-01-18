import torch
from torch import nn
from d2l import torch as d2l

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))
# 通过使用 reduction='none'，你可以获得每个样本的独立损失值而不是对所有样本的损失进行平均或求和。这对于某些特定的训练策略和场景很有用，例如在处理不同样本的权重时，或者在某些样本上放大或减小损失的影响。
loss = nn.CrossEntropyLoss(reduction='none')

trainer = torch.optim.SGD(net.parameters(), lr=0.1) # 这里就包含了w和b
if __name__ == '__main__':
    num_epochs = 10


    # 训练模型
    for epoch in range(num_epochs):
        # X y表示小批量样本的特征和标签
        for X, y in train_iter:
            # 计算损失
            l = loss(net(X), y)
            # 梯度清零 防止梯度累加
            trainer.zero_grad()
            # 计算梯度
            l.sum().backward()
            # 迭代模型参数
            trainer.step()
        with torch.no_grad():
            # 测试模型准确率
            total_acc = 0
            for X,y in test_iter:
                # 计算准确率
                acc = (net(X).argmax(axis=1) == y).sum()
                total_acc += acc
            # 计算平均准确率
            print(f'epoch {epoch + 1}, acc {total_acc / len(test_iter.dataset):.3f}')




