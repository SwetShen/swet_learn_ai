import torch
from torch import nn
import matplotlib.pyplot as plt

# ============== plot ==============
fig = plt.figure(figsize=(10, 4))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
# ============== prepare dataset ==============
x = torch.linspace(0, 2 * torch.pi, 100).unsqueeze(1)
y = torch.sin(x) + torch.normal(0, 0.2, x.shape)

ax1.plot(x.detach().numpy(), y.detach().numpy(), 'ro')
# ============== generate dataset ==============
k = 10
len_sentence = 10
past_list = []
future_list = []
for i in range(x.shape[0] - len_sentence - k):
    past_list.append(y[i:i + len_sentence])
    future_list.append(y[i + k:i + len_sentence + k])

features = torch.cat(past_list, dim=1).permute([1, 0])
labels = torch.cat(future_list, dim=1).permute([1, 0])
# ============== create model ==============
model = nn.Sequential(
    nn.Linear(len_sentence, 20),
    nn.Tanh(),
    nn.Linear(20, len_sentence)
)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
# ============== train ==============
epochs = 1000
for epoch in range(epochs):
    optimizer.zero_grad()
    predict_labels = model(features)
    loss = criterion(predict_labels, labels)
    loss.backward()
    optimizer.step()

    print(f"epoch {epoch + 1}/{epochs} -- loss:{loss.item():.4f}")
# ============== predict ==============
model.eval()
test_features = y.reshape(-1, len_sentence)
results = model(test_features)
ax2.plot(x.detach().numpy(), results.view(-1).detach().numpy(), 'bo')
plt.show()
