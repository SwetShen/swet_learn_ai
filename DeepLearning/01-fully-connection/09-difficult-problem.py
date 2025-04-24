import torch
import matplotlib.pyplot as plt
from torch import nn

noise = 0.2
x = torch.linspace(0, 2, 20).reshape(-1, 1)
y = torch.sin(x ** 2)

plt.plot(x.detach().numpy(), y.detach().numpy(), 'ro')

mode = nn.Sequential(
    nn.Linear(1, 2),
    nn.Sigmoid(),
    nn.Linear(2, 1)
)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(mode.parameters(), lr=0.1)
epochs = 20000
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
