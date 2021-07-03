import math 
from jax import random
import jax.numpy as np
from jax.ops import index, index_update

class Sinusoids():
    """ Regression dataset that implements pattern generation task for SNNs
    """
    def __init__(self, key, seq_length, num_samples, num_inputs, input_freq):
        self.seq_length   = seq_length
        self.num_inputs   = num_inputs
        self.num_samples  = num_samples
        self.freq_list    = np.array([1, 2, 3, 5]) # Hz
        self.dt           = 1e-3 # sec
        self.t            = np.arange(0, seq_length*self.dt, self.dt) 
        self.inp_freq     = input_freq
        self.key          = key

        # Random input spikes
        self.x = (random.uniform(self.key, (self.num_samples, self.num_inputs, self.seq_length)) < self.dt * self.inp_freq).astype(np.float32)
        
        # Randomize output amplitude and phase
        amplitude_list = random.uniform(self.key, (self.num_samples, len(self.freq_list)), minval=0.5, maxval=2)
        phase_list = random.uniform(self.key, (self.num_samples, len(self.freq_list)), minval=0, maxval=2*math.pi)
        
        # Normalized sum of sinusoids
        self.y = np.zeros((self.num_samples, self.seq_length, 1))
        for i in range(self.num_samples):
            summed_sinusoid = sum([amplitude_list[i, ix] * np.sin(2*math.pi*f*self.t + phase_list[i, ix]) for ix, f in enumerate(self.freq_list)])
            summed_sinusoid_norm = summed_sinusoid/np.max(np.abs(summed_sinusoid))
            self.y = index_update(self.y, index[i,:,0], summed_sinusoid_norm)

    def __len__(self):
            return self.num_samples

    def __getitem__(self, idx):
        if not isinstance(idx, np.ndarray):
            idx = np.array(idx)
        return self.x[idx], self.y[idx]
