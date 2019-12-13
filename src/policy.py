import numpy as np
from scipy.special import expit as sigmoid

class BaseFeatureMap():
    """ The feature map phi: R^d x {0,1} -> R^m that maps the feature vector and sensitive attribute of an
    individual into the feature space of the parameters theta"""

    def __init__(self, dim_theta):
        self.dim_theta = dim_theta

    def __call__(self, features):
        return self.map(features)

    def map(self, features):
        raise NotImplementedError("Subclass must override map(x).")

class IdentityFeatureMap(BaseFeatureMap):
    """ The feature map phi as an identity mapping"""

    def __init__(self, dim_theta):
        super(IdentityFeatureMap, self).__init__(dim_theta)

    def map(self, features):
        return features

class LogisticPolicy():
    def __init__(self, dim_theta, num_iterations, batch_size): 
        self.num_iterations = num_iterations
        self.batch_size = batch_size

        self.theta = np.zeros(dim_theta)
        self.feature_map = IdentityFeatureMap(dim_theta)
    
    def __call__(self, x, s):
        features = np.concatenate((x, s), axis=1)
        probability = sigmoid(self.feature_map(features) @ self.theta)
        return np.random.binomial(1, probability)

    def update(self, data, learning_rate):
        X, S, Y = data
        pass