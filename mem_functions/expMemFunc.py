import numpy as np

from .baseMem import BaseMembershipFunction

class ExponentialMembershipFunction(BaseMembershipFunction):
    """
    Models sharp, aggressive decay from a central ideal point.
    Unlike Gaussian curves which are rounded at the top, this forms a sharp spike. 
    Useful for 'Strict Tolerance' contexts, like defining an 'Exact Match' where 
    being even slightly off causes a rapid drop in membership.
    
    Parameters:
        center: The exact point of peak membership (1.0).
        k: The decay rate multiplier. Higher k means a narrower, sharper spike.
    """
    def __init__(self, center, k, universe_min, universe_max):
        super().__init__(universe_min, universe_max)
        self.center = float(center)
        self.k = float(k)

    def evaluate(self, x):
        x = np.asarray(x)
        
        # Symmetrical exponential decay utilizing absolute distance from the center
        # e^(-k * |x - center|)
        return np.exp(-self.k * np.abs(x - self.center))

    def get_label(self):
        return f"Exponential (c={self.center}, k={self.k})"