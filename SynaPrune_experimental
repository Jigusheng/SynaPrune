import torch
from torch import Tensor
import torch.nn as nn

def Devarg(x):
    return x[x.abs().argmax()]

def Norm(x, eps=1e-8):
    return x/max(x.abs().sum(), eps)

def Sim(a, b, eps=1e-8):
    a -= a.mean()
    b -= b.mean()
    return a @ b.transpose(0,1)/max((a.square().sum().sqrt()) * b.square().sum().sqrt(), eps)

class Linear:
    def __init__(self, In, out, f=torch.sigmoid):
        self.In = In
        self.out = out
        self.f = f
        self.W = 2 * torch.rand(In, out) - 1
        self.B = 2 * torch.rand(1, out) - 1
        self.t1_I = torch.zeros(1, In)
    def __call__(self, x):
        self.I = x
        self.O = self.f(x @ self.W + self.B)
        return self.O
    def Update(self, loss, lr):
        loss = Devarg(loss)
        I = self.I
        if Sim(self.I, self.t1_I).abs() >= 0.5:
            I -= self.t1_I
        self.W += lr * loss * Norm(I).transpose(0,1) @ Norm(self.O)
        self.B += lr * loss * Norm(self.O)
    def consolidate(self):
        self.t1_I = self.I

class Last(Linear):
    def __init__(self, In, out, f=torch.sigmoid):
        super().__init__(In, out, f=f)
    def Update(self, loss, lr):
        I = self.I
        if Sim(self.I, self.t1_I).abs() >= 0.5:
            I -= self.t1_I
        self.W += lr * Norm(I).transpose(0,1) @ loss
        self.B += lr * loss

class Sequential:
    def __init__(self):
        self.layers = []
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    def consolidate(self):
        for layer in self.layers:
            layer.consolidate()
    def Update(self, loss, lr):
        for layer in self.layers:
            layer.Update(loss, lr)
