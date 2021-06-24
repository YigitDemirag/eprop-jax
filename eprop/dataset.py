import math
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split

class Sinusoids(Dataset):
    def __init__(self, seed, seq_length, num_samples, num_inputs, input_freq=50):
        self.seq_length   = seq_length
        self.num_inputs   = num_inputs
        self.num_samples  = num_samples
        self.freq_list    = torch.tensor([1, 2, 3, 5]) # Hz
        self.dt           = 1e-3 # s
        self.t            = torch.arange(0, seq_length*self.dt, self.dt) # s
        self.inp_freq     = input_freq

        # Fix seeds
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        # Random input
        self.x = (torch.rand(self.num_samples, self.num_inputs, self.seq_length) < self.dt * self.inp_freq).float()

        # Randomized output amplitude and phase
        amplitude_list = torch.FloatTensor(self.num_samples, len(self.freq_list)).uniform_(0.5, 2)
        phase_list = torch.FloatTensor(self.num_samples, len(self.freq_list)).uniform_(0, 2 * math.pi)

        # Normalized sum of sinusoids
        self.y = torch.zeros(self.num_samples, self.seq_length)
        for i in range(self.num_samples):
          summed_sinusoid = sum([amplitude_list[i, ix] * torch.sin(2*math.pi*f*self.t + phase_list[i, ix]) for ix, f in enumerate(self.freq_list)])
          self.y[i,:] = summed_sinusoid/torch.max(torch.abs(summed_sinusoid))

            

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.x[idx], self.y[idx]
