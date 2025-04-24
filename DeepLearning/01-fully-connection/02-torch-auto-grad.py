import torch

x = torch.tensor([1., 2., 3., 4.], requires_grad=True)
y = x ** 2

torch.sum(y).backward()
print(x.grad)
