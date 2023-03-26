import torch


class TwoHiddenLayerTanhNN(torch.nn.Sequential):
    """Neural network with two hidden layers for nonstationary kernels."""
    def __init__(
        self,
        dim_input: int,
        dim_hidden: int,
        dim_output: int,
        softmax: bool = True,
    ) -> None:
        """

        Parameters
        ----------
        dim_input: int
            Input dimension.
        dim_hidden: int
            Hidden dimension.
        dim_output: int
            Output dimension.
        softmax: bool = True
            Apply softmax on the output or not?

        """
        super().__init__()
        self.add_module("linear1", torch.nn.Linear(dim_input, dim_hidden))
        self.add_module("activation1", torch.nn.Tanh())
        self.add_module("linear2", torch.nn.Linear(dim_hidden, dim_hidden))
        self.add_module("activation2", torch.nn.Tanh())
        self.add_module("linear3", torch.nn.Linear(dim_hidden, dim_output))
        if softmax:
            self.add_module("activation3", torch.nn.Softmax(dim=1))
