import torch
from torch import Tensor
import torch.nn as nn

def Norm(x, eps=1e-8):
    x = x/max(x.abs().sum(), eps)
    return x

def Sim(a, b, eps=1e-8):
    a = a - a.mean()
    b = b - b.mean()
    return (a * b).sum()/max((a.square().sum().sqrt()) * b.square().sum().sqrt(), eps)

def Devarg(x):
    return x[x.abs().argmax()]

class Linear:
    def __init__(self, In ,out, f=torch.sigmoid):
        self.W = 2 * torch.rand(In, out) - 1
        self.B = 2 * torch.rand(1, out) - 1
        self.f = f
        self.t1_I = torch.zeros(1,In)
    def __call__(self, x):
        self.I = x
        self.O = self.f(x @ self.W + self.B)
        return self.O
    def Update(self, loss, lr=1.0):
        loss = Devarg(loss)
        I = self.I
        if Sim(self.I, self.t1_I).abs() >= 0.5:
            I = self.I - self.t1_I
        self.W += lr * loss * Norm(I).transpose(0,1) @ Norm(self.O)
        self.B += lr * loss * Norm(self.O)
    def record(self):
        self.t1_I = self.I
    
class Last(Linear):
    def __init__(self, In, out, f=torch.sigmoid):
        super().__init__(In, out, f=f)
    def Update(self, loss, lr):
        I = self.I
        if Sim(self.I, self.t1_I).abs() >= 0.5:
            I = self.I - self.t1_I
        self.W += lr * Norm(I).transpose(0,1) @ loss
        self.B += lr * loss

class Sequential:
    def __init__(self):
        self.layers = []
    def add(self, layer):
        self.layers.append(layer)
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    def record(self):
        for layer in self.layers:
            layer.record()
    def Update(self, loss, lr):
        for layer in self.layers:
            layer.Update(loss, lr)
    def train(self, data, actual, lr=1.0, epoches=5):
        for e in range(epoches):
            pred = self(data)
            loss = actual - pred
            self.Update(loss, lr=lr)
            self.record()

#Test
data = torch.Tensor([[0,0,1],[0,0,0],[0,1,0],[0,1,1],[1,0,0]])
actual = torch.Tensor([1,0,1,0,1]).view(-1,1)
model = Sequential()
model.add(Linear(3,4))
model.add(Last(4,1))


print("Before training: ", model(data))
for _ in range(100):
    for d in range(data.shape[0]):
        d1, a1 = data[d,:].view(1,-1), actual[d].view(1,-1)
        model.train(d1, a1, epoches=5, lr=1)

print("After training: ", model(data))
