# 我们对回归问题感兴趣 不是问多少二而是问哪一个！！
import torch
from IPython import display
from d2l import torch as d2l

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# 28 * 28 = 784  10个类别
num_inputs = 784
num_outputs = 10

# 初始化模型参数
W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)  # 784 * 10 需要优化
b = torch.zeros(num_outputs, requires_grad=True)  # 10指定了输出的维度


# 定义softmax操作
def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)
    return X_exp / partition  # 这里应用了广播机制


# 定义模型 torch.matmul表示矩阵乘法 X.reshape((-1, W.shape[0]))表示将X转换为一个二维矩阵
def net(X):
    return softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) + b)


# 定义损失函数
def cross_entropy(y_hat, y):
    return -torch.log(y_hat[range(len(y_hat)), y])


# 定义准确率
def accuracy(y_hat, y):
    """计算预测正确的数量"""
    # 返回y_hat中最大值的索引 保证y_hat和y的形状一致 并且存在概率预测的形状
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        # argmax返回最大值的索引
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())
