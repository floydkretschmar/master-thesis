import numpy as np
#pylint: disable=no-name-in-module
from scipy.special import expit as sigmoid

# -------------------------------------------------------------------------
# region Synthetic 1D Distribution for Proof of Concept (Split)
# -------------------------------------------------------------------------
class SplitDistribution():
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

        return x.reshape((n, -1)), s.reshape((n, -1))

    def sample_labels(self, x, s):
        yprob = 0.8 * sigmoid(0.6 * (x + 3)) * sigmoid(
            -5 * (x - 3)
        ) + sigmoid(x - 5)

        return np.random.binomial(1, yprob)