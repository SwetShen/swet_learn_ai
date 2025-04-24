import torch
import matplotlib.pyplot as plt
from torch import nn

noise = 0.2
x = torch.linspace(0, 1, 20).reshape(-1, 1)
y = x ** 2

plt.plot(x.detach().numpy(), y.detach().numpy(), 'ro')

mode = nn.Sequential(
    nn.Linear(1, 1),
    nn.Sigmoid()
)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(mode.parameters(), lr=0.5)

predict_y = mode(x)

line, = plt.plot(x.detach().numpy(), predict_y.detach().numpy(), 'b--')
epochs = 10000
for epoch in range(epochs):
    optimizer.zero_grad()
    predict_y = mode(x)
    loss = criterion(predict_y, y)
    print(loss)
    loss.backward()
    optimizer.step()
    line.set_data(x.detach().numpy(), predict_y.detach().numpy())
    plt.pause(0.1)
