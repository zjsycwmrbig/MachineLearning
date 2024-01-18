import torch
import torchvision
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l


def get_fashion_mnist_labels(labels):
    """返回Fashion-MNIST数据集的文本标签。"""
    text_labels = [
        't-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt',
        'sneaker', 'bag', 'ankle boot']
    # 返回一个列表
    return [text_labels[int(i)] for i in labels]


def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    """Plot a list of images."""
    # figsize 表示图像的大小
    figsize = (num_cols * scale, num_rows * scale)
    # 设置画布
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
    # 将axes转换为列表
    axes = axes.flatten()
    # 遍历所有的图像
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        # 设置不显示坐标轴
        ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        # 如果有标题 则设置标题
        if titles:
            ax.set_title(titles[i])
    return axes


d2l.use_svg_display()

# 下载数据集
trans = transforms.ToTensor()
# root表示数据集下载的路径 train表示是否为训练集 transform表示对数据集进行的操作 download表示是否需要下载
mnist_train = torchvision.datasets.FashionMNIST(root="../data", train=True, transform=trans, download=True)
mnist_test = torchvision.datasets.FashionMNIST(root="../data", train=False, transform=trans, download=True)

print(len(mnist_train), len(mnist_test))
