import numpy as np


class StandardScaler:
    """Standardize features by removing the mean and scaling to unit variance.
    """
    def __init__(self, values: np.ndarray) -> None:
        """

        Parameters
        ----------
        values: np.ndarray, shape=(num_samples, num_dims)
            The data used to compute the per-dimension `mean` and `std` used
            for later scaling.

        """
        if values.ndim != 2:
            raise ValueError("values.shape=(num_samples, num_dims)")
        self.scale = values.std(axis=0, keepdims=True)
        if np.any(self.scale <= 0.0):
            raise ValueError("scale must be positive")
        self.mean = values.mean(axis=0, keepdims=True)

    def preprocess(self, raw: np.ndarray) -> np.ndarray:
        """Pre-process `raw` by removing the mean and scaling to unit
        variance.

        Parameters
        ----------
        raw: np.ndarray, shape=(num_samples, num_dims)
            The raw features to be standardized.

        Returns
        -------
        transformed: np.ndarray, shape=(num_samples, num_dims)
            The standardized features.

        """
        transformed = (raw - self.mean) / self.scale
        return transformed

    def postprocess_mean(self, transformed: np.ndarray) -> np.ndarray:
        """Inverse-transform the predictive mean.

        Parameters
        ----------
        transformed: np.ndarray, shape=(num_samples, num_dims)
            The predictive mean of a probabilistic model built on the
            transformed features.

        Returns
        -------
        raw: np.ndarray, shape=(num_samples, num_dims)
            The corresponding predictive mean using the raw features.

        """
        raw = transformed * self.scale + self.mean
        return raw

    def postprocess_std(self, transformed: np.ndarray) -> np.ndarray:
        """Inverse-transform the predictive standard deviation.

        Parameters
        ----------
        transformed: np.ndarray, shape=(num_samples, num_dims)
            The predictive standard deviation of a probabilistic model
            built on the transformed features.

        Returns
        -------
        raw: np.ndarray, shape=(num_samples, num_dims)
            The corresponding predictive standard deviation using the raw
            features.

        Notes
        -----
        We do not add the `mean` back to the standard deviation.

        """
        raw = transformed * self.scale
        return raw


class MinMaxScaler:
    """Transform features by scaling each feature to a given range.

    This estimator scales and translates each feature individually such
    that it is in the given range on the training set, e.g. between
    -1 and 1.

    The transformation is given by::

        X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
        X_scaled = X_std * (max - min) + min

    where min, max = feature_range.

    Notes
    -----
    This class is adapted from the scikit-learn library:
    https://github.com/scikit-learn/scikit-learn/blob/0d378913be6d7e485b792ea36e9268be31ed52d0/sklearn/preprocessing/_data.py#L265

    """
    def __init__(
            self,
            values: np.ndarray,
            expected_range: tuple = (-1.0, 1.0),
    ) -> None:
        """

        Parameters
        ----------
        values: np.ndarray, shape=(num_samples, num_dims)
            The data used to compute the per-dimension min and max used for
            later scaling.

        expected_range: tuple, default=(-1.0, 1.0)
            Desired range of transformed data.

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

        Parameters
        ----------
        raw: np.ndarray, shape=(num_samples, num_dims)
            The raw values to be transformed.

        Returns
        -------
        transformed: np.ndarray, shape=(num_samples, num_dims)
            The transformed values.

        """
        standardized = (raw - self.data_min) / self.data_ptp
        transformed = standardized * self.ptp + self.min
        return transformed

    def postprocess(self, transformed):
        """Inverse-transform.

        Parameters
        ----------
        transformed: np.ndarray, shape=(num_samples, num_dims)
            Transformed values in the expected range.

        Returns
        -------
        raw: np.ndarray, shape=(num_samples, num_dims)
            Raw values in the original range.

        """
        standardized = (transformed - self.min) / self.ptp
        raw = standardized * self.data_ptp + self.data_min
        return raw
