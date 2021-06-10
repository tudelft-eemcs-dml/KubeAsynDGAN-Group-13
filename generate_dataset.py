import torch, torchvision
import torchvision.datasets as datasets
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision.utils import save_image

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

generate = True
download = True
train_size = 5000
test_size = 1000

# Get Real Dataset
if generate:
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))])

    mnist_trainset = datasets.MNIST(root='./data', train=True, download=download, transform=transform)
    mnist_testset = datasets.MNIST(root='./data', train=False, download=download, transform=transform)

    train_loader = DataLoader(mnist_trainset, batch_size=len(mnist_trainset))
    test_loader = DataLoader(mnist_testset, batch_size=len(mnist_testset))

    train_real = next(iter(train_loader))[0].numpy()[:train_size]
    test_real = next(iter(test_loader))[0].numpy()[:test_size]

    np.save("train_real", train_real)
    np.save("test_real", test_real)

# Generate Fake dataset
# Load latest model
class Generator(nn.Module):
    def __init__(self, g_input_dim, g_output_dim):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(g_input_dim, 256)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features * 2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features * 2)
        self.fc4 = nn.Linear(self.fc3.out_features, g_output_dim)

    # forward method
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.leaky_relu(self.fc3(x), 0.2)
        return torch.tanh(self.fc4(x))

PATH = "generator_model_test"
G = Generator(g_input_dim = 100, g_output_dim = 784).to(device)
G.load_state_dict(torch.load(PATH))
G.eval()

with torch.no_grad():
    train_fake = []
    test_fake = []

    # Train and test loops
    for i in range(train_size):
        z = Variable(torch.randn(1, 100).to(device))
        x_f = G(z).detach().numpy().reshape(1, 28, 28)
        train_fake.append(x_f)

    for i in range(test_size):
        z = Variable(torch.randn(1, 100).to(device))
        x_f = G(z).detach().numpy().reshape(1, 28, 28)
        test_fake.append(x_f)

    # Convert to ndarray and save
    train_fake = np.array(train_fake)
    test_fake = np.array(test_fake)
    np.save("train_fake", train_fake)
    np.save("test_fake", test_fake)