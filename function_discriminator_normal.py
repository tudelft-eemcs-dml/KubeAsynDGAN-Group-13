""" Definition of a KubeML function to train the LeNet network with the MNIST dataset"""
import logging
import random
from typing import List, Any, Union, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torch.optim import SGD

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Discriminator(nn.Module):
    """ Discriminator
    """

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 1024)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features//2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features//2)
        self.fc4 = nn.Linear(self.fc3.out_features, 1)

    def forward(self, x):
        x.view(x.shape[0], -1)
        y = F.leaky_relu(self.fc1(x), 0.2)
        y = F.dropout(y, 0.3)
        y = F.leaky_relu(self.fc2(y), 0.2)
        y = F.dropout(y, 0.3)
        y = F.leaky_relu(self.fc3(y), 0.2)
        y = F.dropout(y, 0.3)
        y = torch.sigmoid(self.fc4(y))
        return y

    def configure_optimizers(self) -> torch.optim.Optimizer:
        # adam = Adam(self.parameters(), lr=self.lr)
        # return adam
        sgd = SGD(self.parameters(), lr=self.lr, momentum=0.9, weight_decay=1e-4)
        return sgd

    def init(self):
        pass

    def train(self, batch, batch_index) -> float:
        # define the device for training and load the data
        criterion = nn.BCELoss()
        total_loss = 0

        x,_ = batch

        bs = batch[0].shape[0]

        self.optimizer.zero_grad()

        # train discriminator on real
        x_real, y_real = x[:, :, 0].view(bs,784), torch.ones(bs, 1)
        x_real, y_real = Variable(x_real.to(device)), Variable(y_real.to(device))
        
        output = self(x_real)
        real_loss = criterion(output, y_real)

        # train discriminator on fake
        x_fake, y_fake = x[:,:,1].view(bs,784), torch.zeros(bs, 1)
        x_fake, y_fake = Variable(x_fake.to(device)), Variable(y_fake.to(device))

        output = self(x_fake)
        fake_loss = criterion(output, y_fake)

        # gradient backprop & optimize ONLY D's parameters
        loss = real_loss + fake_loss
        loss.backward()

        # step with the optimizer
        self.optimizer.step()

        total_loss += loss.data.item()

        if batch_index % 10 == 0:
            logging.info(f"Index {batch_index}, error: {loss.item()}")

        return total_loss

    def validate(self, batch, _) -> Tuple[float, float]:
        criterion = nn.BCELoss()

        x,_ = batch

        bs = batch[0].shape[0]

        test_loss = 0
        correct = 0

        # test discriminator on real
        x_real, y_real = x[:,:,0].view(bs,784), torch.ones(bs, 1)
        x_real, y_real = Variable(x_real.to(device)), Variable(y_real.to(device))

        output = self(x_real)
        real_loss = criterion(output, y_real)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(y_real.view_as(pred)).sum().item()

        # test discriminator on fake
        x_fake, y_fake = x[:,:,1].view(bs,784), torch.zeros(bs, 1)
        x_fake, y_fake = Variable(x_fake.to(device)), Variable(y_fake.to(device))

        output = self(x_fake)
        fake_loss = criterion(output, y_fake)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(y_fake.view_as(pred)).sum().item()

        loss = real_loss + fake_loss
        test_loss += loss.data.item()

        accuracy = 100. * correct / batch[0].shape[0]
        # self.logger.debug(f'accuracy {accuracy}')

        return accuracy, test_loss

    def infer(self, data: List[Any]) -> Union[torch.Tensor, np.ndarray, List[float]]:
        x = torch.tensor(data)
        x = x.view(x.size(0), 784)
        x = Variable(x.to(device))

        output = self(x)

        return output


class MnistDataset(Dataset):

    def __init__(self, data, labels):
        # super().__init__("mnist")
        self.data = data
        self.labels = labels
        self.transf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    def __getitem__(self, index):
        x_real = self.data[index]
        x_fake = self.labels[index]

        return x_real, x_fake

    def __len__(self):
        return len(self.data)

torch.manual_seed(42)
random.seed(42)

x = np.load("x_train_disc.npy")
y = np.load("y_train_disc.npy")
batch_size = 64
train_set = MnistDataset(x, y)
train_loader = DataLoader(train_set, batch_size=batch_size)

discriminator = Discriminator()
discriminator.lr = 0.0002
discriminator.optimizer = discriminator.configure_optimizers()
discriminator.batch_size = batch_size

train = False
if train:
    for i in range(10):
        print("============ Epoch " + str(i + 1) + " ============")
        for i, batch in enumerate(train_loader, start=1):
            image, label = batch
            loss = discriminator.train(batch, i)
            print("Loss: " + str(loss))
            # accuracy, loss = discriminator.validate(batch, i)
            # print("Accuracy: " + str(accuracy) + ", Loss: " + str(loss))

infer = True
if infer:
    output = discriminator.infer(np.load('inference.npy'))
    print(output)