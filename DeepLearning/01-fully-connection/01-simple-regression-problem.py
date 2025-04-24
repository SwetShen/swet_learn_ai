import numpy as np
import matplotlib.pyplot as plt

noise = 0.2
x = np.linspace(0, 1, 20)
y = 3 * x + 2
y += np.random.normal(0, noise, y.shape)

plt.plot(x, y, 'ro')

predict_w = 0.1
predict_b = 0.1
predict_y = predict_w * x + predict_b
line, = plt.plot(x, predict_y, 'b--')
epochs = 10000
for epoch in range(epochs):
    predict_y = predict_w * x + predict_b
    loss = np.sum((predict_y - y) ** 2)
    print(loss)
    predict_w -= np.mean(2 * x * (predict_y - y)) * 0.1
    predict_b -= np.mean(2 * (predict_y - y)) * 0.1
    line.set_data(x, predict_y)
    plt.pause(0.1)
