import numpy as np
import matplotlib.pyplot as plt

from .baseMem import BaseMembershipFunction

class TriangularMembershipFunction(BaseMembershipFunction):
    """
    Defines a Triangular Membership Function (TMF) for fuzzy logic modeling.
    
    This shape is the most widely used in fuzzy systems due to its computational 
    efficiency and simplicity. It represents a concept that peaks at a single 
    ideal point and falls off linearly on both sides (e.g., 'Target Room Temperature').
    
    Parameters:
        a: Lower bound (coordinate where membership starts rising from 0)
        m: Core/Peak (coordinate where membership reaches its absolute maximum of 1)
        b: Upper bound (coordinate where membership returns to 0)
        universe_min: Minimum boundary of the universe of discourse
        universe_max: Maximum boundary of the universe of discourse
    """
    def __init__(self, a, m, b, universe_min, universe_max):
        # Initialize the base class to set up and validate the universe boundaries
        super().__init__(universe_min, universe_max)
        
        # Enforce the strict geometric rule of a triangle: left < peak < right
        if not (a < m < b):
            raise ValueError("Parameters must strictly satisfy: a < m < b for a standard triangle.")
            
        self.a = float(a)
        self.m = float(m)
        self.b = float(b)

    def evaluate(self, x):
        """
        Evaluates the degree of membership for the input tensor 'x'.
        Utilizes vectorized numpy operations to process massive arrays (like neural 
        sensor data or continuous state monitoring) simultaneously without for-loops.
        """
        x = np.asarray(x)
        
        # Calculate the mathematical line equation for the left upward slope
        left_slope = (x - self.a) / (self.m - self.a)
        
        # Calculate the mathematical line equation for the right downward slope
        right_slope = (self.b - x) / (self.b - self.m)
        
        # Combine the slopes: np.minimum handles the intersection forming the peak at 'm'
        intersection = np.minimum(left_slope, right_slope)
        
        # Floor the bottom at 0: np.maximum ensures out-of-bound inputs do not return negative values
        membership_degree = np.maximum(intersection, 0.0)
        
        return membership_degree

    def get_label(self):
        """Generates the descriptive string label used in the UI/plot legend."""
        return f"Triangular (a={self.a}, m={self.m}, b={self.b})"

class TriangularMemberFunction:
    """
    Defines a Triangular Membership Function for fuzzy logic modeling
    Parameters:
        a: Lower bound (membership is 0)
        m: Core/Peak (membership is 1)
        b: Upper bound (membership is 0)
        universe_min: Minimum value of the universe of discourse
        universe_max: Maximum value of the universe of discourse
    """
    def __init__(self, a, m, b, universe_min, universe_max):
        if not (a<m<b):
            raise ValueError("Parameters must satisy: a < m < b for a standard triangle.")
        if universe_min >= universe_max:
            raise ValueError("Universe minimum must be less than maximum.")
        
        self.a = float(a)
        self.m = float(m)
        self.b = float(b)

        self.universe_min = float(universe_min)
        self.universe_max = float(universe_max)

    def evaluate(self, x):
        """
        Evaluates the degree of membership for input x.
        Accepts both signals scaler values and numpy arrays.
        """
        x = np.asarray(x)

        # Calculates the left upward slope
        left_slope = (x - self.a) / (self.m - self.a)
        
        # Calculates the right downward slope
        right_slope = (self.b - x) / (self.b - self.m)

        # Applies the compact mathematical formula: max(min(left, right), 0)
        intersection = np.minimum(left_slope, right_slope)
        membership_degree = np.maximum(intersection, 0.0)
        
        return membership_degree

    def plot(self, title="Triangular Membership Function"):
        """Generates a graph using the object's predefined universe boundaries."""
        # Generate 500 evenly spaced points across the class-level universe
        x = np.linspace(self.universe_min, self.universe_max, 500)
        y = self.evaluate(x)
        
        plt.figure(figsize=(8, 5))
        plt.plot(x, y, label=f"TMF (a={self.a}, m={self.m}, b={self.b})", color='blue', linewidth=2)
        
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel("Universe of Discourse (x)", fontsize=12)
        plt.ylabel("Degree of Membership A(x)", fontsize=12)
        plt.ylim(-0.05, 1.1) 
        
        # X-axis limits are drawn directly from the object parameters
        plt.xlim(self.universe_min, self.universe_max)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(loc="upper right")
        
        plt.show()


if __name__ == "__main__":
    # 1. Initialize the fuzzy set alongside its universe (0 to 12)
    optimal_alertness = TriangularMembershipFunction(a=3, m=6, b=9, universe_min=0, universe_max=12)

    # 2. Draw and show the graph without needing to pass the boundaries again
    optimal_alertness.plot(title="Cognitive State: Optimal Alertness")
