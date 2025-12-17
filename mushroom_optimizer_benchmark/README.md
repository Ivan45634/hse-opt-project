# Mushroom (LIBSVM) optimizer comparison

## 1. Task Description

Binary classification on the LIBSVM mushrooms dataset (categorical features expanded to a high-dimensional real vector via the LIBSVM format). Labels are mapped to $\{0; 1\}$

### Objective

Empirical risk minimization with cross-entropy:
$$
\min_\theta f(\theta)=\frac{1}{n}\sum_{i=1}^n\ell(\theta;x_i,y_i)
$$
where $\ell$ is cross entropy for a 2-clas classifier

### Model

Training classifier is a 2-layer ReLU network (`hidden_dim > 0`)
```python
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
# ...
```

**Key property**: The objective is non-convex and only $L$-smooth if there is no value dropped below 0, due to ReLU. However, it is differentiable almost everywhere.

### Dataset preprocessing

Dataset is splitted into train/val/test with the ratio 7:2:1.

```python
def libsvm_prepocess(args):
    g, seed_worker = utils.set_global_seed(args.seed)
    if args.dataset == "mushrooms":
        if not os.path.exists(f"./{args.data_path}/mushrooms"):
            os.system(f"cd ./{args.data_path}\n wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/mushrooms \n cd ..")
        X, y = load_svmlight_file(f"./{args.data_path}/mushrooms")
        y = y - 1
        X = X.toarray()
# ...
    
    if args.scale:
        A = np.diag(np.exp(np.random.uniform(-args.scale_bound, args.scale_bound, X.shape[1])))
        X = X @ A
# ...
```
## Run the experiment
```python
python scripts/libsvm/run_mushrooms_table1.py
```

## Visualization
 ```python
python scripts/libsvm/plot_mushrooms_table1.py
```
