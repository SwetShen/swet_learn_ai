import torch
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(10, 4))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

noise = 0.2
x = torch.linspace(0, 1, 20)
y = x ** 2

ax1.plot(x, y, 'ro')
ax2.plot(x, y, 'ro')


def sigmoid(x):
    return 1 / (1 + torch.exp(-x))


def tanh(x):
    return (torch.exp(x) - torch.exp(-x)) / (torch.exp(x) + torch.exp(-x))


def relu(x):
    x[x < 0] = 0
    return x


predict_w_s = torch.tensor(0.1, requires_grad=True)
predict_b_s = torch.tensor(0.1, requires_grad=True)
predict_w_t = torch.tensor(0.1, requires_grad=True)
predict_b_t = torch.tensor(0.1, requires_grad=True)
predict_s = sigmoid(predict_w_s * x + predict_b_s)
predict_t = relu(predict_w_t * x + predict_b_t)
line1, = ax1.plot(x.detach().numpy(), predict_s.detach().numpy(), 'b--')
line2, = ax2.plot(x.detach().numpy(), predict_t.detach().numpy(), 'g--')
epochs = 10000
for epoch in range(epochs):
    predict_s = sigmoid(predict_w_s * x + predict_b_s)
    predict_t = relu(predict_w_t * x + predict_b_t)
    loss1 = torch.mean((predict_s - y) ** 2)
    loss2 = torch.mean((predict_t - y) ** 2)
    print(f"loss1:{loss1.item():.4f}  loss2:{loss2.item():.4f}")
    loss1.backward()
    loss2.backward()
    with torch.no_grad():
        predict_w_s -= predict_w_s.grad * 0.5
        predict_b_s -= predict_b_s.grad * 0.5
        predict_w_s.grad.zero_()
        predict_b_s.grad.zero_()
        predict_w_t -= predict_w_t.grad * 0.5
        predict_b_t -= predict_b_t.grad * 0.5
        predict_w_t.grad.zero_()
        predict_b_t.grad.zero_()
    line1.set_data(x.detach().numpy(), predict_s.detach().numpy())
    line2.set_data(x.detach().numpy(), predict_t.detach().numpy())
    plt.pause(0.1)
