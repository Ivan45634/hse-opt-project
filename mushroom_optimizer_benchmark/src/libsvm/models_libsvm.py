import torch
import torch.nn as nn

class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim=3, output_dim=1, w_0 = None, dtype=None):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim, dtype=dtype)
        if w_0 is None:
            nn.init.zeros_(self.linear.weight)
    def forward(self, x):
        y = self.linear(x)
        return y

def zero_init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.zeros_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
            
def uniform_init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)

def zero_uniform_init_weights(m):
    if isinstance(m, nn.Linear) and m.weight.shape[1] == 112:
        nn.init.zeros_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)

class LinearClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=10, output_dim=2, 
                 weight_init="uniform", dtype=None, bias=True):
        super(LinearClassifier, self).__init__()
        if hidden_dim > 0:
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim, dtype=dtype, bias=bias),
                # nn.Dropout(p=0.1),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim, dtype=dtype, bias=bias),
                nn.Softmax(dim=1),
            )
        else:
            self.net = nn.Sequential(
                nn.Linear(input_dim, output_dim, dtype=dtype, bias=bias),
                nn.ReLU(),
                nn.Softmax(dim=1),
            )

        if weight_init == "zeroes": 
            init_fn = zero_init_weights
        elif weight_init == "zero/uniform":
            init_fn = zero_uniform_init_weights
        else:
            init_fn = uniform_init_weights
        self.net.apply(init_fn)
        if weight_init == "bad_scaled":
            with torch.no_grad():
                val_w, val_b = 1e2, 1e2
                for layer in self.net:
                    if hasattr(layer, "weight"):
                        layer.weight.data *= val_w
                        val_w = val_w**(-1)
                    if hasattr(layer, "bias"):
                        layer.weight.data *= val_b
                        val_b = val_b**(-1)

    def forward(self, x):
        out = self.net(x)
        return out
