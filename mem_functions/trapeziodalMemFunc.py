import numpy as np

from .baseMem import BaseMembershipFunction

class TrapezoidalMembershipFunction(BaseMembershipFunction):
    """
    Models concepts that have a 'plateau' of full membership. 
    Ideal for states that maintain peak intensity over a sustained duration 
    (e.g., 'Deep Sleep Phase' in a cognitive tracking AI).
    
    Parameters:
        a: Bottom-left coordinate (starts rising from 0)
        b: Top-left coordinate (reaches 1)
        c: Top-right coordinate (begins falling from 1)
        d: Bottom-right coordinate (returns to 0)
    """
    def __init__(self, a, b, c, d, universe_min, universe_max):
        super().__init__(universe_min, universe_max)
        
        # Enforce strict sequential ordering of the trapezoid's points
        if not (a < b <= c < d):
            raise ValueError("Trapezoid points must strictly progress: a < b <= c < d")
            
        self.a, self.b = float(a), float(b)
        self.c, self.d = float(c), float(d)

    def evaluate(self, x):
        x = np.asarray(x)
        
        # Upward slope calculation
        left_slope = (x - self.a) / (self.b - self.a)
        
        # Downward slope calculation
        right_slope = (self.d - x) / (self.d - self.c)
        
        # The inner np.minimum caps the top at 1.0 (the plateau).
        # The outer np.maximum floors the bottom at 0.0 (ignoring out-of-bounds).
        return np.maximum(np.minimum(np.minimum(left_slope, 1.0), right_slope), 0.0)

    def get_label(self):
        return f"Trapezoidal (a={self.a}, b={self.b}, c={self.c}, d={self.d})"