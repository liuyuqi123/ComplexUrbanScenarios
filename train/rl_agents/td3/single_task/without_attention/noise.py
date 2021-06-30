"""
Normal Noise and OU Noise modified based on OpenAI baselines.
"""

import numpy as np
import scipy.stats as stats


class NormalActionNoise:
    """
    Gaussian Noise.
    """
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def __call__(self):
        return np.random.normal(self.mu, self.sigma)

    def __repr__(self):
        return 'NormalActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)


class OrnsteinUhlenbeckActionNoise:
    """
    OpenAI baselines Ornstein-Uhlenbeck Noise.
    """

    def __init__(self, mu, sigma, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)


class TruncNormalNoise:
    """
    Truncated normal(gaussian) noise.

    Based on scipy.stats.truncnorm.

    Default is 1-dimension.
    """
    def __init__(self,
                 a: float,
                 b: float,
                 mu: float = 0,
                 sigma: float = 1,
                 ):

        self.myclip_a = a
        self.myclip_b = b

        self.mu = mu
        self.sigma = sigma

        self._a, self._b = (self.myclip_a - self.mu) / self.sigma, (self.myclip_b - self.mu) / self.sigma
        self.distribution = stats.truncnorm(self._a, self._b, loc=self.mu, scale=self.sigma)

    def __call__(self):
        # todo add dimension args
        x = self.distribution.rvs()

        return x

    def __repr__(self):
        return 'TruncatedNormalNoise(mu={}, sigma={}, a={}, b={})'\
            .format(self.mu, self.sigma, self.myclip_a, self.myclip_b)

