import torch
from torch import nn

P = torch.tensor([
    [1., 0.],
    [0., 1.],
    [0., 1.],
    [1., 0.],
    [1., 0.]
])

Q = torch.tensor([
    [0.3, 0.7],
    [0.1, 0.9],
    [0.2, 0.8],
    [0.5, 0.5],
    [0.6, 0.4]
])

criterion = nn.CrossEntropyLoss()
loss = criterion(Q, P)
print(loss)


def softmax(x):
    print(torch.exp(x).shape)
    return torch.exp(x) / torch.sum(torch.exp(x), dim=-1)

print(softmax(Q))
# print(torch.mean(torch.sum(-P * torch.log(softmax(Q)), dim=-1)))
