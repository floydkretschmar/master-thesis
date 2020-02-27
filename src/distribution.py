import numpy as np
#pylint: disable=no-name-in-module
from scipy.special import expit as sigmoid
from scipy.stats.distributions import truncnorm
from sklearn.model_selection import train_test_split


####################### DISTRIBUTIONS #######################

class BaseDistribution():
    def __init__(self, bias=False):
        self.bias = bias

    def sample_test_dataset(self, n_test):
        """
        Draws a nxd matrix of non-sensitive feature vectors, a n-dimensional vector of sensitive attributes 
        and a n-dimensional ground truth vector used for testing.

        Args:
            n: The number of examples for which to draw attributes.

        Returns:
            x: nxd matrix of non-sensitive feature vectors
            s: n-dimensional vector of sensitive attributes
        """
        raise NotImplementedError("Subclass must override sample_test_dataset(self, n_test).")    


    def sample_train_dataset(self, n_train):
        """
        Draws a nxd matrix of non-sensitive feature vectors, a n-dimensional vector of sensitive attributes 
        and a n-dimensional ground truth vector used for training.

        Args:
            n: The number of examples for which to draw attributes.

        Returns:
            x: nxd matrix of non-sensitive feature vectors
            s: n-dimensional vector of sensitive attributes
        """
        raise NotImplementedError("Subclass must override sample_train_dataset(self, n_train).")    


class GenerativeDistribution(BaseDistribution):
    def __init__(self, fraction_protected, bias=False):
        super(GenerativeDistribution, self).__init__(bias)

        self.fraction_protected = fraction_protected

    def sample_features(self, n):
        """
        Draws both a nxd matrix of non-sensitive feature vectors, as well as a n-dimensional vector
        of sensitive attributes.

        Args:
            n: The number of examples for which to draw attributes.

        Returns:
            x: nxd matrix of non-sensitive feature vectors
            s: n-dimensional vector of sensitive attributes
        """
        raise NotImplementedError("Subclass must override sample_features(self, n).")    

    def sample_labels(self, x, s):
        """
        Draws a n-dimensional ground truth vector.

        Args:
            x: nxd matrix of non-sensitive feature vectors
            s: n-dimensional vector of sensitive attributes

        Returns:
            y: n-dimensional ground truth vector
        """
        raise NotImplementedError("Subclass must override sample_labels(self, x, s).")    

    def sample_train_dataset(self, n_train):
        x, s = self.sample_features(n_train)
        y = self.sample_labels(x, s)

        return x, s, y

    def sample_test_dataset(self, n_test):
        return self.sample_train_dataset(n_test)


class SplitDistribution(GenerativeDistribution):
    def __init__(self, fraction_protected, bias=False):
        super(SplitDistribution, self).__init__(fraction_protected, bias)

    def sample_features(self, n):
        s = (
            np.random.rand(n, 1) < self.fraction_protected
        ).astype(int)
        x = 3.5 * np.random.randn(n, 1) + 3 * (0.5 - s)

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
        

class UncalibratedScore(GenerativeDistribution):
    """An distribution modelling an uncalibrated score."""

    def __init__(self, fraction_protected, bias=False):
        super(UncalibratedScore, self).__init__(fraction_protected, bias)
        self.bound = 0.8
        self.width = 30.0
        self.height = 3.0
        self.shift = 0.1
        self.bias = bias

    def _pdf(self, x):
        """Get the probability of repayment."""
        num = (
            np.tan(x)
            + np.tan(self.bound)
            + self.height
            * np.exp(-self.width * (x - self.bound - self.shift) ** 4)
        )
        den = 2 * np.tan(self.bound) + self.height
        return num / den

    def sample_features(self, n):
        s = (
            np.random.rand(n, 1) < self.fraction_protected
        ).astype(int)

        shifts = s - 0.5
        x = truncnorm.rvs(
            -self.bound + shifts, self.bound + shifts, loc=-shifts
        ).reshape(-1, 1)

        if self.bias:
            ones = np.ones((n, 1))
            x = np.hstack((ones, x))

        return x, s

    def sample_labels(self, x, s):
        if self.bias:
            x = x[:,1]

        yprob = self._pdf(x)        
        return np.expand_dims(np.random.binomial(1, yprob), axis=1)


class ResamplingDistribution(BaseDistribution):
    """Resample from a finite dataset."""

    def __init__(self, dataset, test_percentage, bias=False):
        super(ResamplingDistribution, self).__init__(bias)
        x, s, y = dataset
        self.x, self.x_test, self.y, self.y_test, self.s, self.s_test = train_test_split(x, y, s, test_size=test_percentage)

        self.total_test_samples = self.x_test.shape[0]
        self.test_sample_indices = np.arange(self.total_test_samples)

        self.total_training_samples = self.x.shape[0]
        self.training_sample_indices = np.arange(self.total_training_samples)

        if self.bias:
            self.x = np.hstack([np.ones([self.total_training_samples, 1]), self.x])
            self.x_test = np.hstack(
                [np.ones([self.x_test.shape[0], 1]), self.x_test]
            )
        self.feature_dim = self.x.shape[1]

    def sample_train_dataset(self, n_train):
        n = min(self.total_training_samples, n_train)
        indices = np.random.choice(self.training_sample_indices, n, replace=True)

        return self.x[indices].reshape((n_train, -1)), self.y[indices].reshape((n_train, -1)), self.s[indices].reshape((n_train, -1))

    def sample_test_dataset(self, n_test=None):
        n = min(self.total_test_samples, n_test) if n_test is not None else self.total_test_samples
        indices = np.random.choice(self.test_sample_indices, n, replace=True)

        return self.x_test[indices].reshape((n_test, -1)), self.y_test[indices].reshape((n_test, -1)), self.s_test[indices].reshape((n_test, -1))