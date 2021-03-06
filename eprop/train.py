from jax import random
from dataset import Sinusoids
from utils import initialize_parameters, mse_loss
from model import SRNN

def train(seed, epochs, n_inp, n_rec, n_out, tau_rec, tau_out, 
          lr_inp, lr_rec, lr_out, w_gain, thr, n_t, gamma, reg, f0, dt):
    
    # deterministic JAX 
    key = random.PRNGKey(seed)

    # create dataset
    toy_dataset = Sinusoids(key, seq_length=n_t, num_samples=4, num_inputs=n_inp, input_freq=50)
    
    # create network
    theta = initialize_parameters(key, n_inp, n_rec, n_out, w_gain)
    srnn =  SRNN(n_inp, n_rec, n_out, tau_rec, tau_out, thr,
                 gamma, lr_inp, lr_rec, lr_out, n_t, reg, f0, dt)
    
    # train
    loss_arr = []
    for epoch in range(epochs):
        for x,y in zip(toy_dataset.x, toy_dataset.y):

            yhat, traces, reg_term = srnn.forward(theta, x)
            loss   = yhat-y # Scaled derivative of (yhat-y)^2
            grads  = srnn.acc_gradient(loss, traces, reg_term, theta)
            theta  = srnn.upd_weights(theta, grads)
        
        loss_arr.append(mse_loss(yhat, y))
        if epoch%10 == 0:
            print(f'Epoch: [{epoch}/{epochs}] - MSE Loss: {mse_loss(yhat, y):.4f}')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--n_inp', type=int, default=100)
    parser.add_argument('--n_rec', type=int, default=100)
    parser.add_argument('--n_out', type=float, default=1)
    parser.add_argument('--tau_rec', type=float, default=30e-3)
    parser.add_argument('--tau_out', type=float, default=30e-3)
    parser.add_argument('--lr_inp', type=float, default=2e-5)
    parser.add_argument('--lr_rec', type=float, default=5e-5)
    parser.add_argument('--lr_out', type=float, default=2e-5)
    parser.add_argument('--w_gain', type=float, default=0.01)
    parser.add_argument('--thr', type=float, default=0.1)
    parser.add_argument('--n_t', type=int, default=1000)
    parser.add_argument('--gamma', type=float, default=0.3)
    parser.add_argument('--reg', type=float, default=1e-4)
    parser.add_argument('--f0', type=float, default=12)
    parser.add_argument('--dt', type=float, default=1e-3)

    args = parser.parse_args()

    train(seed=args.seed,
          epochs=args.epochs,
          n_inp=args.n_inp,
          n_rec=args.n_rec,
          n_out=args.n_out,
          tau_rec=args.tau_rec,
          tau_out=args.tau_out,
          lr_inp=args.lr_inp,
          lr_rec=args.lr_rec,
          lr_out=args.lr_out,
          w_gain=args.w_gain,
          thr=args.thr,
          n_t=args.n_t,
          gamma=args.gamma,
          reg=args.reg,
          f0=args.f0,
          dt=args.dt)