import numpy as np
#pylint: disable=no-name-in-module
from scipy.special import expit as sigmoid
from scipy.stats.distributions import truncnorm

# -------------------------------------------------------------------------
# region Synthetic 1D Distribution for Proof of Concept (Split)
# -------------------------------------------------------------------------
class SplitDistribution():
    def __init__(self, bias=False):
        self.bias = bias

    def sample_features(self, n, fraction_protected):
        """
        Draw examples only for the features of the true distribution.

        Args:
            n: The number of examples to draw.

        Returns:
            x: np.ndarray with the features of dimension (n, k), where k is
                either 1 or 2 depending on whether a constant is added
        """
        s = (
            np.random.rand(n, 1) < fraction_protected
        ).astype(int)
        x = 3.5 * np.random.randn(n, 1) + 3 * (s - 0.5)

        if self.bias:
            ones = np.ones((n, 1))
            x = np.hstack((ones, x))

        return x, s 

    def sample_labels(self, x, s):
        if self.bias:
            x = x[:,1]

        yprob = 0.8 * sigmoid(0.6 * (x + 3)) * sigmoid(
            -5 * (x - 3)
        ) + sigmoid(x - 5)

        return np.expand_dims(np.random.binomial(1, yprob), axis=1)

    def sample_dataset(self, n, fraction_protected):
        x, s = self.sample_features(n, fraction_protected)
        y = self.sample_labels(x, s)

        return x, s, y

# -------------------------------------------------------------------------
# region A score based distribution that is uncalibrated
# -------------------------------------------------------------------------
class UncalibratedScore():
    """An distribution modelling an uncalibrated score."""

    def __init__(self, bias=False):
        super().__init__()
        self.bound = 0.8
        self.width = 30.0
        self.height = 3.0
        self.shift = 0.1
        self.bias = bias

    def pdf(self, x):
        """Get the probability of repayment."""
        num = (
            np.tan(x)
            + np.tan(self.bound)
            + self.height
            * np.exp(-self.width * (x - self.bound - self.shift) ** 4)
        )
        den = 2 * np.tan(self.bound) + self.height
        return num / den

    def sample_features(self, n, fraction_protected):
        s = (
            np.random.rand(n, 1) < fraction_protected
        ).astype(int)

        shifts = s - 0.5
        x = truncnorm.rvs(
            -self.bound - shifts, self.bound - shifts, loc=shifts
        ).reshape(-1, 1)

        if self.bias:
            ones = np.ones((n, 1))
            x = np.hstack((ones, x))

        return x, s

    def sample_labels(self, x, s):
        if self.bias:
            x = x[:,1]

        yprob = self.pdf(x)        
        return np.expand_dims(np.random.binomial(1, yprob), axis=1)

    def sample_dataset(self, n, fraction_protected):
        x, s = self.sample_features(n, fraction_protected)
        y = self.sample_labels(x, s)

        return x, s, y