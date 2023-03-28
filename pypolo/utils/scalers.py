import numpy as np


class StandardScaler:

    def __init__(self, values: np.ndarray) -> None:
        r"""Removing the mean and scaling to unit variance.

        Args:
            values (np.ndarray): The array of shape (num_inputs, dim_inputs)
                used to compute the per-dimension `mean` and `std` used for
                later scaling.

        Raises:
            ValueError: If `values` does not have shape
                (num_inputs, dim_inputs) or if `scale` is not positive.

        """
        if values.ndim != 2:
            raise ValueError(
                "values must be two-dimensional: (num_inputs, dim_inputs)")
        self.scale = values.std(axis=0, keepdims=True)
        if np.any(self.scale <= 0.0):
            raise ValueError("scale must be positive")
        self.mean = values.mean(axis=0, keepdims=True)

    def preprocess(self, raw: np.ndarray) -> np.ndarray:
        r"""Pre-processes `raw` by removing the mean and scaling to unit
        variance.

        Args:
            raw (np.ndarray): The raw features to be standardized.
                Must have shape (num_inputs, dim_inputs).

        Returns:
            np.ndarray: The standardized features with the same shape.

        """
        transformed = (raw - self.mean) / self.scale
        return transformed

    def postprocess_mean(self, transformed: np.ndarray) -> np.ndarray:
        r"""Inverse-transforms the predictive mean.

        Args:
            transformed (np.ndarray): The predictive mean of a probabilistic
                model built on the transformed features.

        Returns:
            np.ndarray: The original predictive mean.
        """
        raw = transformed * self.scale + self.mean
        return raw

    def postprocess_std(self, transformed: np.ndarray) -> np.ndarray:
        """Inverse-transforms the predictive standard deviation.

        Args:
            transformed (np.ndarray): The predictive standard deviation of a
                probabilistic model built on the transformed features.

        Returns:
            np.ndarray: The original predictive standard deviation.

        ??? note

            We should not add the `mean` back to the standard deviation.

        """
        raw = transformed * self.scale
        return raw


class MinMaxScaler:

    def __init__(
        self, values: np.ndarray, expected_range: tuple = (-1.0, 1.0)) -> None:
        r"""Scaling each feature to a given range.

        Args:
            values (np.ndarray): The data used to compute the per-dimension
                `min` and `max` used for later scaling.
                Shape (num_inputs, dim_inputs).
            expected_range (tuple, optional): Desired range of transformed data.
                Shape (min, max). Default is (-1.0, 1.0).

        Raises:
            ValueError: If the `expected_range` or the data range is not valid.

        Attributes:
            min (float): The minimum value of the expected range.
            max (float): The maximum value of the expected range.
            ptp (float): The difference between the maximum and minimum values
                of the expected range. ptp stands for "peak to peak".
            data_min (np.ndarray): The minimum values of the data used for
                scaling. Shape (1, dim_inputs).
            data_max (np.ndarray): The maximum values of the data used for
                scaling. Shape (1, dim_inputs).
            data_ptp (np.ndarray): The differences between the maximum and
                minimum values of the data used for scaling.
                Shape (1, dim_inputs).

        ??? note "Transformation"

            ```python
            data_min, data_max = values.min(axis=0), values.max(axis=0)
            data_ptp = data_max - data_min
            min, max = expected_range
            ptp = max - min
            after = ((before - data_min) / data_ptp) * ptp + min
            ```

        ??? note "Code Authorship Attribution"

            This class is adapted from the
            [scikit-learn](https://github.com/scikit-learn/scikit-learn/blob
            /0d378913be6d7e485b792ea36e9268be31ed52d0/sklearn/preprocessing/
            _data.py#L265) library.

        """
        self.min = expected_range[0]
        self.max = expected_range[1]
        # `ptp` is the acronym for ‘peak to peak’.
        self.ptp = expected_range[1] - expected_range[0]
        if self.ptp <= 0.0:
            raise ValueError("Expected range must be positive.")
        self.data_min = values.min(axis=0, keepdims=True)
        self.data_max = values.max(axis=0, keepdims=True)
        self.data_ptp = self.data_max - self.data_min
        if np.any(self.data_ptp <= 0.0):
            raise ValueError("Data range must be positive.")

    def preprocess(self, raw: np.ndarray) -> np.ndarray:
        """Pre-process `raw` to the expected range.

        Args:
            raw (np.ndarray): The raw values to be transformed.
                Shape (num_inputs, dim_inputs).

        Returns:
            np.ndarray: The transformed values.
                Shape (num_inputs, num_inputs).

        """
        standardized = (raw - self.data_min) / self.data_ptp
        transformed = standardized * self.ptp + self.min
        return transformed

    def postprocess(self, transformed: np.ndarray) -> np.ndarray:
        """Inverse-transform.

        Args:
            transformed (np.ndarray): Transformed values in the expected range.
                Shape (num_inputs, dim_inputs).

        Returns:
            np.ndarray: Raw values in the original range.
                Shape (num_inputs, dim_inputs).

        """
        standardized = (transformed - self.min) / self.ptp
        raw = standardized * self.data_ptp + self.data_min
        return raw
