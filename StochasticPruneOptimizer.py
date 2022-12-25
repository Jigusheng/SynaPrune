def Norm(x, eps=1e-10, dim=None):
    '''
    x = torch.Tensor([[1,0,1]])
    print(Norm(x))
    >>> tensor([[0.5000, 0.0000, 0.5000]])
    '''
    '''This is a normalizing function to normalize the inputs and outputs used in updation of weights'''
    return x.div(x.abs().sum(dim=dim))
  
def Devarg(x):
  '''Returns a value with the largest magnitude'''
    return x.flatten()[x.abs().argmax()]

  
  
#Defining the custom Dense class in pytorch
class Dense(nn.Module):
    def __init__(self, I, O, f=torch.sigmoid):
        super().__init__()
        self.W = nn.Parameter(2 * torch.rand(I, O) - 1, requires_grad=False) #Defining parameters in pytorch Parameters for easier manipulation of parameters although grads are not used
        self.B = nn.Parameter(2 * torch.rand(1, O) - 1, requires_grad=False)
        self.In = I
        self.Out = O 
        self.f = f

    def forward(self,  x):
        self.I = x                 #Save 'x' as a global variable for later use in updation of parameters
        self.O = self.f(x @ self.W + self.B)
        return self.O
    def Update(self, loss):
        loss1 = Devarg(loss) # loss1 over here is an array with only 1 element!
        with torch.no_grad():
            dw = self.I.transpose(0,1) @ self.O
            db = self.O
            dw_lr = F.softmax(dw)
            dw_lr = 1/dw_lr.mean().exp()
            db_lr = F.softmax(db)
            db_lr = 1/db_lr.mean().exp()            
            I = self.I - self.I.mean()
            O = self.O - self.O.mean()
            dw = loss1 * dw_lr * (Norm(I.transpose(0,1)) @ Norm(O))
            db = loss1 * db_lr * Norm(O)
            
           
                
            
            self.Wn = dw
            self.Bn = db

    def clear_n(self, lr=1.0):
        self.W = nn.Parameter(self.W + lr * self.Wn)
        self.B = nn.Parameter(self.B + lr * self.Bn)
        self.Wn = torch.zeros_like(self.Wn)
        self.Bn = torch.zeros_like(self.Bn)
            

class Last(Dense):
    def __init__(self, I, O, f=torch.sigmoid):
        super().__init__(I, O, f=f) #Inherits Dense class but updation of parameters are different for the last layer in a feedforward network.

    
    def Update(self, loss): #loss over here is an Array with 1 or more elements!
        with torch.no_grad():
            dw = self.I.transpose(0,1) @ self.O
            db = self.O
            dw_lr = F.softmax(dw)
            dw_lr = 1/dw_lr.mean()
            db_lr = F.softmax(db)
            db_lr = 1/db_lr.mean()
            I = self.I - self.I.mean()
            O = self.O - self.O.mean()
            dw = dw_lr * (Norm(I.transpose(0,1)) @ loss)
            db = db_lr * loss * Norm(O)
            

            self.Wn = dw
            self.Bn = db
            
            
def train(model, data, actual, epoches=100, lr=1):
    (N1, n_I) = data.shape #N is batch_size, n_I is number of inputs
    (N2, n_O) = actual.shape # n_O is number of outputs
    
    
    for _ in range(epoches):
        for data1, actual1 in zip(data, actual):
    
            d1 = data1.reshape(1, n_I)
            a1 = actual1.reshape(1, n_O)
            
            pred = model(d1)
            loss = loss_fn(a1, pred) 
            model.Update(loss)
            
            print(Devarg(loss))
      
            model.clear_n(lr=lr)
      
'''Hi, pls be kind and not steal my work [beg]'''
