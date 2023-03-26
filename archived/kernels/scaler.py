import torch


class Scaler(torch.nn.Module):
    """
    Source: GPyTorch
    Scale the input data so that it lies in between the lower and upper bounds.

    In training (`self.train()`), this module adjusts the scaling factor to
    the minibatch of data.
    During evaluation (`self.eval()`), this module uses the scaling factor
    from the previous minibatch of data.


    Example:
        >>> train_x = torch.randn(10, 5)
        >>> module = gpytorch.utils.grid.ScaleToBounds(lower_bound=-1.,
        >>>                                            upper_bound=1.)
        >>> module.train()
        >>> # Data should be between -0.95 and 0.95
        >>> scaled_train_x = module(train_x)
        >>> module.eval()
        >>> test_x = torch.randn(10, 5)
        >>> scaled_test_x = module(test_x)  # Scaling is based on train_x
    """
    def __init__(self, lower_bound, upper_bound):
        """

        Parameters
        ----------
        lower_bound: float
            Lower bound of scaled data
        upper_bound: float
            Upper bound of scaled data

        """
        super().__init__()
        self.lower_bound = float(lower_bound)
        self.upper_bound = float(upper_bound)
        self.register_buffer("min_val", torch.tensor(lower_bound))
        self.register_buffer("max_val", torch.tensor(upper_bound))

    def forward(self, x):
        if self.training:
            min_val = x.min()
            max_val = x.max()
            self.min_val.data = min_val
            self.max_val.data = max_val
        else:
            min_val = self.min_val
            max_val = self.max_val
            # Clamp extreme values
            x = x.clamp(min_val, max_val)

        diff = max_val - min_val
        x = (x - min_val) * (0.95 * (self.upper_bound - self.lower_bound) /
                             diff) + 0.95 * self.lower_bound
        return x
