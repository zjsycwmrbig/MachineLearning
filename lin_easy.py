import torch
from torch.utils import data
from torch import nn


def synthetic_data(w, b, num_examples):
    """Generate y = Xw + b + noise."""
    #  normal 正态分布 是指均值为0 方差为1的正态分布
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    # -1 指的是根据第二个参数来确定 保证了y是一个列向量
    return X, y.reshape((-1, 1))


def load_array(data_arrays, data_batch_size, is_train=True):
    """Construct a PyTorch data iterator."""
    dataset = data.TensorDataset(*data_arrays)
    # 通过随机读取来获取batch_size大小的数据 is_train表示是否为训练集 如果是训练集则打乱数据
    return data.DataLoader(dataset, data_batch_size, shuffle=is_train)


if __name__ == '__main__':
    # 生成数据集
    true_w = torch.tensor([2, -3.4])
    true_b = 4.2
    features, labels = synthetic_data(true_w, true_b, 1000)
    batch_size = 10
    data_iter = load_array((features, labels), batch_size)
    # 读取并打印第一个小批量数据样本
    # print(next(iter(data_iter)))

    # 第一个参数表示输入特征的维度 第二个参数表示输出的维度 net的输入是X之类的特征
    net = nn.Sequential(nn.Linear(2, 1))
    # 初始化模型参数 net[0]表示获取net中的第一个模型 weight和bias是随机初始化的 weight 是正态分布的 bias是0
    net[0].weight.data.normal_(0, 0.01)
    net[0].bias.data.fill_(0)

    # 定义损失函数 MSELoss 表示均方损失函数
    loss = nn.MSELoss()

    # 定义优化算法 SGD 是随机梯度下降算法 lr表示学习率 0.03表示每次迭代的学习率 第一个参数表示需要优化的参数
    trainer = torch.optim.SGD(net.parameters(), lr=0.03)

    # 训练模型
    num_epochs = 3
    for epoch in range(num_epochs):
        # X y表示小批量样本的特征和标签
        for X, y in data_iter:
            # 计算损失
            l = loss(net(X), y)
            # 梯度清零 防止梯度累加
            trainer.zero_grad()
            # 计算梯度
            l.backward()
            # 迭代模型参数
            trainer.step()
        # 计算每个epoch的损失 用于观察损失的下降
        l = loss(net(features), labels)
        print(f'epoch {epoch + 1}, loss {l:f}')

    # 比较学到的模型参数和真实的模型参数
    w = net[0].weight.data
    print('w的估计误差：', true_w - w.reshape(true_w.shape))
    b = net[0].bias.data
    print('b的估计误差：', true_b - b)
