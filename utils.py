import jax.numpy as np
from jax import random, jit

def initialize_parameters(key, n_inp, n_rec, n_out, w_gain):
    theta = {}
    theta['inp'] = random.normal(key, (n_rec, n_inp)) * w_gain
    theta['rec'] = random.normal(key, (n_rec, n_rec)) * w_gain
    theta['out'] = random.normal(key, (n_out, n_rec)) * w_gain
    theta['fb']  = random.normal(key, (n_out, n_rec)) * w_gain
    return theta

@jit
def mse_loss(x,y):
    return np.mean(np.power(x-y, 2))
