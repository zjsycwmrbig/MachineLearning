import torch
from torch import nn
import time

def train_ch6(net, train_iter, test_iter, num_epochs, lr):
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)

    net.apply(init_weights)
    device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('training on', device)
    net.to(device)
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)

    for epoch in range(num_epochs):
        start = time.time()
        for X, y in train_iter:
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()

        with torch.no_grad():
            total_count = 0
            total_correct = 0
            for X, y in test_iter:
                X, y = X.to(device), y.to(device)
                y_hat = net(X)
                l = loss(y_hat, y)
                total_count += l.sum()
                total_correct += (y_hat.argmax(axis=1) == y).sum()
            acc = total_correct / total_count
            print('epoch', epoch + 1, 'loss', total_count, 'acc', acc, 'time', time.time() - start)


