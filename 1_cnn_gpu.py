from __future__ import print_function

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

class CnnNet(nn.Module):

    def __init__(self):
        super(CnnNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6,
                               kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(6, 16, 5, 1)

        self.fc1 = nn.Linear(in_features=16*5*5, out_features=120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Activation funtion: ReLu
        x = F.relu(self.conv1(x))
        # Max pooling over a (2, 2) window for subsampling
        x = F.max_pool2d(x, kernel_size=(2, 2))
        # If the size is a square we can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), kernel_size=2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # Output layer
        x = self.fc3(x)
        return x
    
    def num_flat_features(self, x):
        num_features = 1
        one_x_size = x.size()[1:]   # all dimensions except the batch dimension
        # print(one_x_size)
        for dim in one_x_size:
            num_features *= dim
        # print(num_features)
        return num_features

def load_data():
    batch_size = 128
    num_workers = 4
    transform = transforms.Compose(transforms=[
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset=trainset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=num_workers)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(dataset=testset,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=num_workers)
    return train_loader, test_loader


def train(train_loader):
    cnn_net = CnnNet().cuda()
    print(cnn_net)
    
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(params=cnn_net.parameters(), lr=0.01, momentum=0.9)
    # optimizer = optim.Adagrad(cnn_net.parameters(), lr=0.01, lr_decay=0.7)
    optimizer = optim.Adam(cnn_net.parameters(), lr=0.001, weight_decay=0.005)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)
    epoch_num = 100
    for epoch in range(epoch_num):
        running_loss = 0.0
        
        for i, data in enumerate(train_loader, 0):
            images, labels = data
            # wrap them in Variable
            inputs, labels = Variable(images.cuda()), Variable(labels.cuda())
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward
            outputs = cnn_net(inputs)
            # backward
            loss = criterion(outputs, labels)
            loss.backward()
            # optimiz: step() can be called once the gradients are computed using backward().
            optimizer.step()
            # scheduler.step()
            # print statistics
            running_loss += loss.data[0]
            if (epoch % 10 == 9) and ( i % 100 == 99):
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0
    print('Finished Training')
    return cnn_net


def test(cnn_net, test_loader):
    total, correct = 0, 0
    for data in test_loader:
        imgs, labels = data
        outputs = cnn_net(Variable(imgs.cuda()))
        _, predicted = torch.max(outputs.data, 1)
        predicted = predicted.cpu()
        total += labels.size(0)
        correct += (predicted == labels).sum()
    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))


def main():
    train_loader, test_loader = load_data()
    cnn_net = train(train_loader)
    test(cnn_net, test_loader)

if __name__ == "__main__":
    main()
