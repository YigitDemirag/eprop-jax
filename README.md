# eprop-jax

A JAX re-implementation of [e-prop](https://www.nature.com/articles/s41467-020-17236-y) algorithm. eprop-jax is written to be simple, clean and fast.
I tried to replicate the pattern generation task in `eprop/dataset.py`, as described in the paper. And the implementation `eprop/model.py` includes only Leaky Integrate-and-Fire (LIF) neurons.

### Example Usage

Since the implementation is JIT compiled, I suggest doing experimentation/modifications on a Jupyter notebook e.g. `regression_task.py`, to avoid XLA compilation at every run.

It's just enough to run `python eprop/train.py` to test the network with default hyperparameters. 

```
$ python eprop/train.py
Epoch: [0/100] - MSE Loss: 0.3905
Epoch: [10/100] - MSE Loss: 0.2088
Epoch: [20/100] - MSE Loss: 0.0821
Epoch: [30/100] - MSE Loss: 0.0395
Epoch: [40/100] - MSE Loss: 0.0307
Epoch: [50/100] - MSE Loss: 0.0265
Epoch: [60/100] - MSE Loss: 0.0187
Epoch: [70/100] - MSE Loss: 0.0221
Epoch: [80/100] - MSE Loss: 0.0173
Epoch: [90/100] - MSE Loss: 0.0217
```
### License

MIT
