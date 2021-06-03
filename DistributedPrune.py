import torch
import nnlib as nn # A module with activation functions 
from tqdm import tqdm

class Dense:
    def __init__(self, In, out, f=nn.linear):
        self.In, self.out, self.f = In, out, f
        self.W = torch.nn.Parameter(2 * torch.rand((In, out)) - 1, requires_grad=True)
        self.B = torch.nn.Parameter(2 * torch.rand((1, out)) - 1, requires_grad=True)
    def __call__(self, x):
        self.W.grad = None
        self.B.grad = None
        self.I = x
        self.O = self.f(torch.matmul(x, self.W) + self.B)
        return self.O
    def Update(self, loss, loss1, learning_rate):
        Local  = nn.Standardize(self.O)
        self.O.backward(torch.ones(self.O.shape), retain_graph=True)
        with torch.no_grad():
            self.W += learning_rate * loss1 * nn.Standardize(self.W.grad) * Local
            self.B += learning_rate * loss1 * nn.Standardize(self.B.grad) * Local
    
class Last(Dense):
    def __init__(self, In, out, f=nn.linear):
        super().__init__(In, out, f=f)
        self.name = "(Last) Layer"

    def Update(self, loss, loss1, learning_rate):
        self.O.backward(torch.ones(self.O.shape), retain_graph=True)
        with torch.no_grad():
            Local = nn.Standardize(self.O)
            self.W += learning_rate * nn.Standardize(self.W.grad) * Local * loss
            self.B += learning_rate * nn.Standardize(self.B.grad) * Local * loss

class Sequential:
    def __init__(self):
        self.Layers = []

    def add(self, layer):
        self.Layers.append(layer)

    def __call__(self, x):
        for layer in self.Layers:
            x = layer(x)
        return x

    def compile(self, Loss=nn.Simpleloss):
        self.Loss = Loss

    def train(self, X_train, Y_train, X_test, Y_test, n_epoch=1, learning_rate=1):
        for e in tqdm(range(n_epoch)):
            nn.clear()
            for i in range(X_train.shape[0]):      
                Y_pred = self(X_train[i:i+1])
                with torch.no_grad():
                    loss = self.Loss(Y_train[i:i+1], Y_pred)
                    loss1 = nn.Devarg(loss)
                for layer in self.Layers:
                    layer.Update(loss, loss1, learning_rate)
                print("Training Loss: ", loss1, "Validation loss: ", )

            print("Training Loss: ", loss1, "Validation loss: ", )
