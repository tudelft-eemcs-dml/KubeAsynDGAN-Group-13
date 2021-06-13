import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import json 
import subprocess, os
import re
import threading
from redisai import Client
import time
my_env = os.environ.copy()
my_env["KUBECONFIG"] = os.path.expanduser(f"~/.kube/config")

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# job 
job = '71e87af8'

# todo kill this process
def port_forward(pod):
    subprocess.run("kubectl -n kubeml port-forward " + redispod + " 6379:6379", env=my_env, shell=True)
    
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

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(784, 1024)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features//2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features//2)
        self.fc4 = nn.Linear(self.fc3.out_features, 1)
    
    # forward method
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = F.dropout(x, 0.3)
        return torch.sigmoid(self.fc4(x))

# Load latest generator model
PATH = "generator_model_test"
G = Generator(g_input_dim = 100, g_output_dim = 784).to(device)
G.load_state_dict(torch.load(PATH))
G.eval()

criterion = nn.BCELoss()
lr = 0.0002
G_optimizer = optim.Adam(G.parameters(), lr = lr)

G_losses = []
bs = 64

# port forward to redisAI
out = subprocess.check_output("kubectl -n kubeml get pods", env=my_env, shell=True)
redispod = re.search('redis-([^\s]+)', out.decode("utf-8")).group(0)

print("Port forwarding to pod" + redispod)
th = threading.Thread(target=port_forward, args=(redispod,))
th.start()

print("sleep 5 seconds to get the stuff running")
time.sleep(5)
con = Client(host='localhost', port=6379)

print("=== GETTING LATEST MODEL ===")
fc1_weight = con.tensorget(job + ':fc1.weight')
fc1_bias = con.tensorget(job + ':fc1.bias')
fc2_weight = con.tensorget(job + ':fc2.weight')
fc2_bias = con.tensorget(job + ':fc2.bias')
fc3_weight = con.tensorget(job + ':fc3.weight')
fc3_bias = con.tensorget(job + ':fc3.bias')
fc4_weight = con.tensorget(job + ':fc4.weight')
fc4_bias = con.tensorget(job + ':fc4.bias')

D_latest = Discriminator()

with torch.no_grad():    
    D_latest = Discriminator().to(device)
    D_latest.fc1.weight = nn.Parameter(torch.tensor(fc1_weight))
    D_latest.fc1.bias = nn.Parameter(torch.tensor(fc1_bias))
    D_latest.fc2.weight = nn.Parameter(torch.tensor(fc2_weight))
    D_latest.fc2.bias = nn.Parameter(torch.tensor(fc2_bias))
    D_latest.fc3.weight = nn.Parameter(torch.tensor(fc3_weight))
    D_latest.fc3.bias = nn.Parameter(torch.tensor(fc3_bias))
    D_latest.fc4.weight = nn.Parameter(torch.tensor(fc4_weight))
    D_latest.fc4.bias = nn.Parameter(torch.tensor(fc4_bias))

n_epochs = 10

train_dataset = torch.from_numpy(np.load('x_train_disc.npy'))
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=bs, shuffle=True)

for epoch in range(n_epochs):
    G_losses = []
    print("===== EPOCH " + str(epoch + 1) + " =====")
    # TODO remove this loader stuff
    for batch_idx, x in enumerate(train_loader):
        x_f = x[:, :, 1] # get fake images
        G.zero_grad()

        z = Variable(torch.randn(x_f.size(0), 100).to(device))
        y = Variable(torch.ones(x_f.size(0), 1).to(device))
        G_output = G(z)

        D_output = D_latest(G_output)

        G_loss = criterion(D_output, y)

        # # gradient backprop & optimize ONLY G's parameters
        G_loss.backward()
        G_optimizer.step()
        G_losses.append(G_loss.data.item())

    print("Loss: " + str(torch.mean(torch.FloatTensor(G_losses))))

# Saving new model
# torch.save(G.state_dict(), PATH)