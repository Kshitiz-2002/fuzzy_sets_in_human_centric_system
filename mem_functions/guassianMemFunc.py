import numpy as np

from .baseMem import BaseMembershipFunction

class GaussianMembershipFunction(BaseMembershipFunction):
    """
    Models natural, symmetrical phenomena with no strict outer boundaries (never truly hits 0).
    Highly favored in neural networks for modeling 'Normal/Baseline States' because 
    it provides smooth, differentiable gradients everywhere.
    
    Parameters:
        center: The focal point where membership is exactly 1.
        sigma: The standard deviation (width/spread of the bell curve).
    """
    def __init__(self, center, sigma, universe_min, universe_max):
        super().__init__(universe_min, universe_max)
        self.center = float(center)
        self.sigma = float(sigma)

    def evaluate(self, x):
        x = np.asarray(x)
        
        # Standard Gaussian bell curve mathematical formulation
        # e^(-0.5 * ((x - center) / sigma)^2)
        return np.exp(-0.5 * ((x - self.center) / self.sigma)**2)

    def get_label(self):
        return f"Gaussian (c={self.center}, σ={self.sigma})"