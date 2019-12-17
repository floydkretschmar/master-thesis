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
