import torch
import matplotlib.pyplot as plt
from torch import nn

noise = 0.4
x = torch.linspace(0, 2, 20).reshape(-1, 1)
y = 3 * x + 2
y += torch.normal(0, noise, y.shape)

plt.plot(x.detach().numpy(), y.detach().numpy(), 'ro')

mode = nn.Sequential(
    nn.Linear(1, 10),
    nn.Tanh(),
    nn.Linear(10, 10),
    nn.Tanh(),
    nn.Linear(10, 1)
)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(mode.parameters(), lr=0.1)
epochs = 10000
for epoch in range(epochs):
    optimizer.zero_grad()
    predict_y = mode(x)
    loss = criterion(predict_y, y)
    print(loss)
    loss.backward()
    optimizer.step()

predict_y = mode(x)
plt.plot(x.detach().numpy(), predict_y.detach().numpy(), 'b--')
plt.show()
