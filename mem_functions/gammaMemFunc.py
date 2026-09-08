import numpy as np

from .baseMem import BaseMembershipFunction

class GammaMembershipFunction(BaseMembershipFunction):
    """
    Models threshold-based escalations. 
    Ideal for concepts that are completely absent until a specific trigger point is hit, 
    after which they rapidly approach full saturation (e.g., 'Panic Response').
    
    Parameters:
        a: The strict absolute threshold. Anything <= a is exactly 0.
        k: The growth rate multiplier. Higher k means a steeper escalation.
    """
    def __init__(self, a, k, universe_min, universe_max):
        super().__init__(universe_min, universe_max)
        self.a = float(a)
        self.k = float(k)

    def evaluate(self, x):
        x = np.asarray(x)
        
        # Calculate the exponential growth curve: 1 - e^(-k * (x-a)^2)
        growth_curve = 1.0 - np.exp(-self.k * (x - self.a)**2)
        
        # np.where enforces the hard boundary: if x <= a, output 0; otherwise use the curve
        return np.where(x <= self.a, 0.0, growth_curve)

    def get_label(self):
        return f"Gamma (a={self.a}, k={self.k})"