import random

import numpy as np
import torch

import pypolo


def seed_everything(seed):
    random.seed(seed)
    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)
    print(f"Set random seed to {seed} in random, numpy, and torch.")
    return rng


def print_metrics(model, evaluator):
    print(f"Data: {model.num_train:04d} | " +
          f"SMSE: {evaluator.smses[-1]:.4f} | " +
          f"MSLL: {evaluator.mslls[-1]:.4f} | " +
          f"NLPD: {evaluator.nlpds[-1]:.4f} | " +
          f"RMSE: {evaluator.rmses[-1]:.4f} | " +
          f"MAE : {evaluator.maes[-1]:.4f}")


def get_kernel(args):
    """Returns the kernel specified in args.kernel."""
    if args.kernel == "rbf":
        kernel = pypolo.kernels.RBF(
            amplitude=args.init_amplitude,
            lengthscale=args.init_lengthscale,
        )
    elif args.kernel == "ak":
        kernel = pypolo.kernels.AK(
            amplitude=args.init_amplitude,
            lengthscales=np.linspace(
                args.min_lengthscale,
                args.max_lengthscale,
                args.dim_output,
            ),
            dim_input=args.dim_input,
            dim_hidden=args.dim_hidden,
            dim_output=args.dim_output,
        )
    elif args.kernel == "gibbs":
        kernel = pypolo.kernels.Gibbs(
            amplitude=args.init_amplitude,
            dim_input=args.dim_input,
            dim_hidden=args.dim_hidden,
        )
    elif args.kernel == "dkl":
        kernel = pypolo.kernels.DKL(
            amplitude=args.init_amplitude,
            lengthscale=args.init_lengthscale,
            dim_input=args.dim_input,
            dim_hidden=args.dim_hidden,
            dim_output=args.dim_output,
        )
    else:
        raise ValueError(f"Kernel {args.kernel} is not supported.")
    return kernel
