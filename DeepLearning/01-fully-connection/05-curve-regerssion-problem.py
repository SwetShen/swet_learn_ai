import torch
import matplotlib.pyplot as plt

noise = 0.2
x = torch.linspace(0, 1, 20)
y = x ** 2

plt.plot(x, y, 'ro')

predict_w = torch.tensor(0.1, requires_grad=True)
predict_b = torch.tensor(0.1, requires_grad=True)
predict_y = predict_w * x + predict_b
line, = plt.plot(x.detach().numpy(), predict_y.detach().numpy(), 'b--')
epochs = 10000
for epoch in range(epochs):
    predict_y = predict_w * x + predict_b
    loss = torch.mean((predict_y - y) ** 2)
    print(loss)
    loss.backward()
    with torch.no_grad():
        predict_w -= predict_w.grad * 0.1
        predict_b -= predict_b.grad * 0.1
        predict_w.grad.zero_()
        predict_b.grad.zero_()
    line.set_data(x.detach().numpy(), predict_y.detach().numpy())
    plt.pause(0.1)
