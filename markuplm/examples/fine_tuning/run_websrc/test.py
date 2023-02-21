import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
import torch.nn.init as init
import numpy as np
from scipy.optimize import minimize
import time

# Define the neural network architecture
input_size = 50
hidden_size = 50
output_size = 1

# Generate random data for the demo
def DBP(x):
    w1 = np.random.randn(input_size, hidden_size)
    b1 = np.random.randn(hidden_size)
    w2 = np.random.randn(hidden_size, output_size)
    b2 = np.random.randn(output_size)
    
    return np.matmul(np.max(np.matmul(x, w1) + b1, 0), w2) + b2

x = np.random.randn(600, input_size)
y = DBP(x)




# Define the PyTorch model
class TwoLayerNet(torch.nn.Module):
    def __init__(self):
        super(TwoLayerNet, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

# Define the loss function
def loss_function(theta, x, y):
    # Convert the flattened PyTorch tensor into a tensor with the correct shape
    # theta = torch.from_numpy(theta.reshape(-1, 1)).float()
    theta = torch.from_numpy(theta).float()

    # print (theta.shape)
    # print (theta[:input_size*hidden_size].reshape(hidden_size, input_size).shape)

    # Update the PyTorch model's parameters with the new values
    model.load_state_dict({'linear1.weight': theta[:input_size*hidden_size].reshape(hidden_size, input_size),
                           'linear1.bias': theta[input_size*hidden_size:input_size*hidden_size+hidden_size],
                           'linear2.weight': theta[input_size*hidden_size+hidden_size:input_size*hidden_size+hidden_size+hidden_size*output_size].reshape(output_size, hidden_size),
                           'linear2.bias': theta[-output_size:]})
    
    # Forward pass through the model to get the predictions
    y_pred = model(torch.from_numpy(x).float())

    # Compute the mean squared error loss
    loss = torch.mean((y_pred - torch.from_numpy(y).float()) ** 2)
    
    return loss.item()

# Initialize the PyTorch model
model = TwoLayerNet()


# Get the initial parameters as a flattened PyTorch tensor
theta_0 = np.concatenate([model.linear1.weight.detach().numpy().reshape(-1),
                          model.linear1.bias.detach().numpy(),
                          model.linear2.weight.detach().numpy().reshape(-1),
                          model.linear2.bias.detach().numpy()])


# Train the model using scipy.optimize.minimize with the nelder-mead method
timer = time.time()
result = minimize(loss_function, theta_0, args=(x, y), method='nelder-mead', options={'maxiter': 30000, 'adaptive': True})
print (time.time() - timer)

# Print the results
# print("Optimization result:")
# print(result)
print("\nFinal loss:", result.fun)
sys.stdout.flush()

print ('*' * 20)
# 重新初始化参数
for m in model.modules():
    if isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight)
        init.constant_(m.bias ,0)

x = torch.tensor(x, dtype=torch.float)
y = torch.tensor(y, dtype=torch.float)

loss_fn = nn.MSELoss()

with torch.no_grad():
    pred = model(x)
    loss = loss_fn(pred, y)
print ('\nLoss before train:', loss.item())

# train with SGD

optimizer = AdamW(model.parameters(), lr=0.005)
for _ in range(500):
    optimizer.zero_grad()
    pred = model(x)
    loss = loss_fn(pred, y)
    loss.backward()
    optimizer.step()

with torch.no_grad():
    pred = model(x)
    loss = loss_fn(pred, y)

print ("\nFinal loss:", loss.item())