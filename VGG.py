import torch
from torch import nn
from d2l import torch as d2l
from Utils import train_ch6

def vgg_block(num_convs, in_channels, out_channels):
    ''' num_convs: 卷积层的数量 in_channels: 输入通道数 out_channels: 输出通道数 '''
    layers = []
    # 第一个卷积层是输入通道数到输出通道数 后面的是一个a*a的卷积
    for _ in range(num_convs):
        layers.append(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        )
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)


def vgg(conv_arch):
    ''' conv_arch: 卷积层的数量 和 输出通道数 是一个pair '''
    conv_blks = []
    in_channels = 1
    for (num_convs, out_channels) in conv_arch:
        conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
        in_channels = out_channels
    return nn.Sequential(
        *conv_blks, nn.Flatten(),
        nn.Linear(out_channels * 7 * 7, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 10)
    )


conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))
radio = 4
small_conv_arch = [(pair[0], pair[1] // radio) for pair in conv_arch]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = vgg(small_conv_arch).to(device)

lr, num_epochs, batch_size = 0.05, 10, 64
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)


def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)


if __name__ == '__main__':

    print(torch.cuda.is_available())
