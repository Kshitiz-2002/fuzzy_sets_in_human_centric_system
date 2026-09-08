import numpy as np

from .baseMem import BaseMembershipFunction

class SShapedMembershipFunction(BaseMembershipFunction):
    """
    Models gradual, smooth transitions between two distinct states without hard angles.
    Ideal for fluid human concepts like 'Transitioning from Youth to Adulthood' or 
    'Gradual Loss of Focus'.
    
    Parameters:
        a: The point where the curve gently departs from 0.
        b: The point where the curve smoothly arrives at 1.
    """
    def __init__(self, a, b, universe_min, universe_max):
        super().__init__(universe_min, universe_max)
        if a >= b:
            raise ValueError("Parameter 'a' (start) must be less than 'b' (end)")
        self.a = float(a)
        self.b = float(b)

    def evaluate(self, x):
        x = np.asarray(x)
        
        # Find the inflection point (exact middle of the S-curve)
        m = (self.a + self.b) / 2.0
        
        # Initialize an empty array of zeros matching the input shape
        y = np.zeros_like(x, dtype=float)
        
        # Calculate the lower convex curve (accelerating upwards)
        idx_lower = (x > self.a) & (x <= m)
        y[idx_lower] = 2.0 * ((x[idx_lower] - self.a) / (self.b - self.a))**2
        
        # Calculate the upper concave curve (decelerating towards 1)
        idx_upper = (x > m) & (x < self.b)
        y[idx_upper] = 1.0 - 2.0 * ((x[idx_upper] - self.b) / (self.b - self.a))**2
        
        # Cap all values past the boundary at exactly 1.0
        y[x >= self.b] = 1.0
        
        return y

    def get_label(self):
        return f"S-Shaped (a={self.a}, b={self.b})"