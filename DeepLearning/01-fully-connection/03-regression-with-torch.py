import torch
import matplotlib.pyplot as plt

noise = 0.2
x = torch.linspace(0, 1, 20)
y = 3 * x + 2
y += torch.normal(0, noise, y.shape)

plt.plot(x, y, 'ro')

predict_w = 0.1
predict_b = 0.1
predict_y = predict_w * x + predict_b
line, = plt.plot(x.detach().numpy(), predict_y.detach().numpy(), 'b--')
epochs = 10000
for epoch in range(epochs):
    predict_y = predict_w * x + predict_b
    loss = torch.sum((predict_y - y) ** 2)
    print(loss)
    predict_w -= torch.mean(2 * x * (predict_y - y)) * 0.1
    predict_b -= torch.mean(2 * (predict_y - y)) * 0.1
    line.set_data(x.detach().numpy(), predict_y.detach().numpy())
    plt.pause(0.1)
