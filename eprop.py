from functools import partial

from einops import rearrange, reduce, repeat

import jax.numpy as np
from jax.lax import scan
from jax import grad, jit, vmap
from jax.ops import index, index_add, index_update
from jax.scipy.signal import correlate

class LSNN():
    def __init__(self, n_inp, n_rec, n_out, tau_rec, tau_out, thr,
                 gamma, lr_inp, lr_rec, lr_out, n_t, reg, f0, dt):
      
        self.n_inp  = n_inp
        self.n_rec  = n_rec
        self.n_out  = n_out
        self.thr    = thr
        self.gamma  = gamma
        self.lr_inp = lr_inp
        self.lr_rec = lr_rec
        self.lr_out = lr_out
        self.n_t    = n_t
        self.reg    = reg
        self.f0     = f0
        self.dt     = dt
        self.alpha  = np.exp(-dt/tau_rec)
        self.kappa  = np.exp(-dt/tau_out)
       
        # Pre-comp
        self.alpha_conv = np.array([self.alpha ** (self.n_t - i - 1) for i in range(self.n_t)]).astype(float) # 1, 1, n_t
        self.kappa_conv = np.array([self.kappa ** (self.n_t - i - 1) for i in range(self.n_t)]).astype(float) # 1, 1, n_t

    @partial(jit, static_argnums=(0,))
    def calc_inp_trace(self, x, h):
        ''' Optimal implementation of e-trace per inp
            correlate in JAX/scipy is conv1d in PyTorch/TF.
            See https://discuss.pytorch.org/t/numpy-convolve-and-conv1d-in-pytorch/12172/4
        '''
        trace_in = repeat(vmap(correlate, in_axes=(0, None))(x, self.alpha_conv)[:,0:self.n_t], 'i t -> r i t', r=self.n_rec) # in, t
        trace_in = np.einsum('tr,rit->rit', h, trace_in)  # n_r, inp_dim, n_t
        trace_in = vmap(correlate, in_axes=(0, None))(trace_in.reshape(self.n_inp*self.n_rec, self.n_t), self.kappa_conv)[:,0:self.n_t].reshape(self.n_rec, self.n_inp, self.n_t)
        return trace_in

    @partial(jit, static_argnums=(0,))
    def calc_rec_trace(self, z, h):
        trace_rec = repeat(vmap(correlate, in_axes=(0, None))(z.T, self.alpha_conv)[:,0:self.n_t], 'i t -> r i t', r=self.n_rec) # in, t
        trace_rec = np.einsum('tr,rit->rit', h, trace_rec) # n_r, inp_dim, n_t
        trace_rec = vmap(correlate, in_axes=(0, None))(trace_rec.reshape(self.n_rec*self.n_rec, self.n_t), self.kappa_conv)[:,0:self.n_t].reshape(self.n_rec, self.n_rec, self.n_t)
        return trace_rec

    @partial(jit, static_argnums=(0,))
    def calc_out_trace(self, z):
        trace_out = vmap(correlate, in_axes=(0, None))(z.T, self.kappa_conv)[:,0:self.n_t]
        return trace_out

    @partial(jit, static_argnums=(0,))
    def calc_fr(self, z):
        fr = np.sum(z, axis=(0)) / (self.n_t * self.dt) 
        reg_term = fr - self.f0
        return reg_term
    
    @partial(jit, static_argnums=(0,))
    def pseudo_der(self, v):
        return self.gamma * np.maximum(np.zeros_like(v), 1 - np.abs((v-self.thr)/self.thr))
    
    @partial(jit, static_argnums=(0,))
    def forward(self, theta, x):

        # Reset diagonal
        rec_weight = (- theta['rec'] * (np.eye(self.n_rec) - 1)).T
        inp_weight = theta['inp'].T
        out_weight = theta['out'].T
        
        def f(carry, x):
            v_curr   = carry[0]
            z_curr   = carry[1]
            vo_curr  = carry[2]
            not_init = carry[3]

            v_next  = (self.alpha * v_curr + np.matmul(z_curr, rec_weight) + np.matmul(x, inp_weight) - z_curr * self.thr) * not_init
            z_next  = (v_next > self.thr).astype(np.float32) * not_init
            vo_next = (self.kappa * vo_curr + np.matmul(z_next, out_weight) ) * not_init

            carry = [v_next, z_next, vo_next,  True]
            y = [v_next, z_next, vo_next]
            return carry, y

        _, (v,z,vo) = scan(f, [np.zeros((self.n_rec)), np.zeros((self.n_rec)), np.zeros((self.n_out)), False], x.T)

        # Pseudo-derivative
        h = self.pseudo_der(v) # nt, nb, nrec

        # E-trace calculation
        traces = {}
        traces['inp'] = self.calc_inp_trace(x, h)
        traces['rec'] = self.calc_rec_trace(z, h)
        traces['out'] = self.calc_out_trace(z)
        
        # Calc firing rate
        reg_term = self.calc_fr(z)
        return vo, traces, reg_term
    
    @partial(jit, static_argnums=(0,))
    def acc_gradient(self, err, traces, reg_term, theta):
        L_loss = np.einsum('t o, o r -> r t', err, theta['fb'])
        L_reg  = repeat(reg_term, 'r -> r (t) ', t=self.n_t)

        L = L_loss + self.reg * L_reg

        grads = {}
        grads['inp']  =   np.clip(-100, np.sum(np.einsum('xt,xyt->xyt', L, traces['inp']), axis=2), 100)
        grads['rec']  =   np.clip(-100, np.sum(np.einsum('xt,xyt->xyt', L, traces['rec']), axis=2), 100)
        grads['out']  =   np.clip(-100, np.einsum('to,rt->or', err, traces['out']), 100)
        return grads
    
    @partial(jit, static_argnums=(0,))
    def upd_weights(self, theta, grads):
        theta['inp'] = np.clip(-1, theta['inp'] - self.lr_inp * grads['inp'], 1)
        theta['rec'] = np.clip(-1, theta['rec'] - self.lr_rec * grads['rec'], 1)
        theta['out'] = np.clip(-1, theta['out'] - self.lr_out * grads['out'], 1)
        return theta
