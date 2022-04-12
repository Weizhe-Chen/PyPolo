from pathlib import Path
import numpy as np


class Logger:
    """Save all the variables for visualization."""
    def __init__(self, eval_outputs: np.ndarray) -> None:
        """

        Parameters
        ----------
        eval_outpus: np.ndarray, shape=(num_x * num_y, 1)
            Outputs for evaluation.

        """
        self.eval_outputs = eval_outputs
        self.means = []
        self.stds = []
        self.errors = []
        self.xs = []
        self.ys = []
        self.nums = []
        self.goals = []

    def append(
        self,
        mean: np.ndarray,
        std: np.ndarray,
        error: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        num: int,
    ):
        """Append the given data.

        Parameters
        ----------
        mean: np.ndarray, shape=(num_samples,)
            Predictive mean vector.
        std: np.ndarray, shape=(num_samples,)
            Predictive standard deviation vector.
        error: np.ndarray, shape=(num_samples,)
            Absolute error vector.
        x: np.ndarray, shape=(num_samples, num_dims)
            Training inputs.
        y: np.ndarray, shape=(num_samples,)
            Training outputs.
        num: int
            Number of training data.

        """
        self.means.append(mean)
        self.stds.append(std)
        self.errors.append(error)
        self.xs.append(x)
        self.ys.append(y)
        self.nums.append(num)

    def save(self, save_dir: str) -> None:
        """

        Parameters
        ----------
        save_dir: str
            Directory to save.

        """
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        means = np.hstack(self.means)
        stds = np.hstack(self.stds)
        errors = np.hstack(self.errors)
        xs = np.vstack(self.xs)
        ys = np.vstack(self.ys)
        nums = np.array(self.nums)
        np.savez_compressed(
            f"{save_dir}/log.npz",
            eval_outputs=self.eval_outputs,
            means=means,
            stds=stds,
            errors=errors,
            xs=xs,
            ys=ys,
            nums=nums,
        )
        print(f"Saved log.npz to {save_dir}")
