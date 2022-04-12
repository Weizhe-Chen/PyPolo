from pathlib import Path
from typing import List, Tuple

import numpy as np

from ..models import IModel
from ..sensors import ISensor


class Evaluator:
    """Evaluate the performance of a model."""

    def __init__(
        self,
        sensor: ISensor,
        task_extent: List[float],
        eval_grid: List[int],
    ) -> None:
        """

        Parameters
        ----------
        sensor: ISensor
            A sensor for getting ground-truth values.
        task_extent: List[float], [xmin, xmax, ymin, ymax]
            Extent of the sampling task.
        eval_grid: List[float], [num_x, num_y]
            Number of samples in the evaluation grid along x and y.

        """
        self.task_extent = task_extent
        self.eval_grid = eval_grid
        self.setup_eval_inputs_and_outputs(sensor)
        self.log2pi = np.log(2 * np.pi)
        self.nums = []
        self.smses = []
        self.mslls = []
        self.nlpds = []
        self.rmses = []
        self.maes = []

    def setup_eval_inputs_and_outputs(self, sensor: ISensor) -> None:
        """Set attributes `eval_inputs` and `eval_outputs`.

        Parameters
        ----------
        sensor: ISensor
            A sensor for getting ground-truth values.

        Attributes
        ----------
        eval_inputs: np.ndarray, shape=(num_x * num_y, 2)
            Inputs for evaluation.
        eval_outpus: np.ndarray, shape=(num_x * num_y, 1)
            Outputs for evaluation.

        """
        xmin, xmax, ymin, ymax = self.task_extent
        num_x, num_y = self.eval_grid
        x = np.linspace(xmin, xmax, num_x)
        y = np.linspace(ymin, ymax, num_y)
        xx, yy = np.meshgrid(x, y)
        self.eval_inputs = np.column_stack((xx.ravel(), yy.ravel()))
        self.eval_outputs = sensor.sense(self.eval_inputs).reshape(-1, 1)

    def eval_prediction(
        self,
        model: IModel,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate prediction performance.

        Parameters
        ----------
        model: IModel
            Probabilistic model.

        Returns
        -------
        mean: np.ndarray, shape=(num_x * num_y, 1)
            Predictive mean.
        std: np.ndarray, shape=(num_x * num_y, 1)
            Predictive standard deviation.
        error: np.ndarray, shape=(num_x * num_y, 1)
            Absolute error.

        """
        _, y_train = model.get_data()
        mean, std = model(self.eval_inputs)
        error = np.fabs(mean - self.eval_outputs)
        self.nums.append(model.num_train)
        self.smses.append(self.calc_smse(error))
        self.mslls.append(self.calc_msll(error, std, y_train))
        self.nlpds.append(self.calc_nlpd(error, std))
        self.rmses.append(self.calc_rmse(error))
        self.maes.append(self.calc_mae(error))
        return mean, std, error

    def calc_log_loss(self, error: np.ndarray, std: np.ndarray) -> np.ndarray:
        """Calculate negative log predictive density.

        Parameters
        ----------
        error: np.ndarray, shape=(num_x * num_y, 1)
            Absolute error.
        std: np.ndarray, shape=(num_x * num_y, 1)
            Predictive standard deviation.

        Returns
        -------
        log_loss: np.ndarray, shape=(num_x * num_y, 1)
            negative log predictive density.

        """
        log_loss = 0.5 * self.log2pi + np.log(std) + 0.5 * np.square(
            error / std)
        return log_loss

    def calc_nlpd(self, error: np.ndarray, std: np.ndarray) -> np.float64:
        """Calculate mean negative log predictive density.

        Parameters
        ----------
        error: np.ndarray, shape=(num_x * num_y, 1)
            Absolute error.
        std: np.ndarray, shape=(num_x * num_y, 1)
            Predictive standard deviation.

        Returns
        -------
        nlpd: np.float64
            Mean negative log predictive density.

        """
        nlpd = np.mean(self.calc_log_loss(error, std))
        return nlpd

    def calc_rmse(self, error: np.ndarray) -> np.float64:
        """Calculate root mean squared error.

        Parameters
        ----------
        error: np.ndarray, shape=(num_x * num_y, 1)
            Absolute error.

        Returns
        -------
        rmse: np.float64
            Root mean squared error.

        """
        rmse = np.sqrt(np.mean(np.square(error)))
        return rmse

    def calc_mae(self, error: np.ndarray) -> np.float64:
        """Calculate mean absolute error.

        Parameters
        ----------
        error: np.ndarray, shape=(num_x * num_y, 1)
            Absolute error.

        Returns
        -------
        mae: np.float64
            Mean absolute error.

        """
        mae = np.mean(error)
        return mae

    def calc_smse(self, error: np.ndarray) -> np.float64:
        """Calculate standardized mean squared error.

        Parameters
        ----------
        error: np.ndarray, shape=(num_x * num_y, 1)
            Absolute error.

        Returns
        -------
        rmse: np.float64
            Standardized mean squared error.

        """
        mse = np.mean(np.square(error))
        smse = mse / self.eval_outputs.var()
        return smse

    def calc_msll(
        self,
        error: np.ndarray,
        std: np.ndarray,
        y_train: np.ndarray,
    ) -> np.float64:
        """Calculate mean standardized log loss.

        Parameters
        ----------
        error: np.ndarray, shape=(num_x * num_y, 1)
            Absolute error.
        std: np.ndarray, shape=(num_x * num_y, 1)
            Predictive standard deviation.
        y_train: np.ndarray, shap=(num_train, 1)
            Training targets.

        Returns
        -------
        msll: np.float64
            Mean standardized log predictive density.

        """
        log_loss = self.calc_log_loss(error, std)
        baseline = self.calc_log_loss(self.eval_outputs - y_train.mean(),
                                      y_train.std())
        msll = np.mean(log_loss - baseline)
        return msll

    def save(self, save_dir: str) -> None:
        """Save all the metrics to the given output directory.

        Parameters
        ----------
        save_dir: str
            Directory to save file.

        """
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        table = np.column_stack((
            self.nums,
            self.smses,
            self.mslls,
            self.nlpds,
            self.rmses,
            self.maes,
        ))
        np.savetxt(
            f"{save_dir}/metrics.csv",
            table,
            fmt="%.8f",
            delimiter=',',
            header="samples,smse,msll,nlpd,rmse,mae",
        )
        print(f"Saved metrics.csv to {save_dir}")
