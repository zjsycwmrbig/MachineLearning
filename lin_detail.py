import torch
import random
from d2l import torch as d2l


def synthetic_data(w, b, num_examples):
    """Generate y = Xw + b + noise."""
    #  normal 正态分布 是指均值为0 方差为1的正态分布
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    # -1 指的是根据第二个参数来确定 保证了y是一个列向量
    return X, y.reshape((-1, 1))


def data_iter(batch_size, features, labels):
    """Iterate through a dataset."""
    num_examples = len(features)
    # 产生一个随机索引
    indices = list(range(num_examples))
    random.shuffle(indices)

    for i in range(0, num_examples, batch_size):
        # 从打乱的索引中取出batch_size个索引
        batch_indices = torch.tensor(
            indices[i:min(i + batch_size, num_examples)])
        # 根据索引取出对应的特征和标签
        yield features[batch_indices], labels[batch_indices]


def linreg(X, w, b):
    """The linear regression model."""
    return torch.matmul(X, w) + b


def squared_loss(y_hat, y):
    """Squared loss."""
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2


def sgd(params, lr, batch_size):
    """Minibatch stochastic gradient descent."""
    with torch.no_grad():
        for param in params:
            # param -= lr * param.grad / batch_size
            param -= lr * param.grad
            param.grad.zero_()


if __name__ == '__main__':
    # 生成数据集
    w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    features, labels = synthetic_data(w, b, 1000)
    batch_size = 10
    lr = 0.03
    num_epochs = 3
    net = linreg
    loss = squared_loss
    for epoch in range(num_epochs):
        for X, y in data_iter(batch_size, features, labels):
            l = loss(net(X, w, b), y)  # Minibatch loss in `X` and `y`
            # Compute gradient on `l` with respect to [`w`, `b`]
            l.sum().backward()
            sgd([w, b], lr, batch_size)  # Update parameters using their gradient

        with torch.no_grad():
            train_l = loss(net(features, w, b), labels)
            print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')
