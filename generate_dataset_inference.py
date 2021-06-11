import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

batch_size = 64

with torch.no_grad():
    x_fake = []

    # Train and test loops
    for i in range(batch_size):
        z = Variable(torch.randn(1, 100).to(device))
        x_f = G(z).detach().numpy().reshape(1, 28, 28)
        x_fake.append(x_f)

    np.save('inference.npy',x_fake)