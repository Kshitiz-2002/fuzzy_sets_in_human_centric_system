import numpy as np
import matplotlib.pyplot as plt

class BaseMembershipFunction:
    """
    Abstract base class for all fuzzy membership functions.
    
    This class standardizes how an AI system interacts with fuzzy sets by 
    enforcing a strict universe of discourse (the absolute boundaries of 
    the concept being modeled) and providing universal visualization tools.
    """
    def __init__(self, universe_min, universe_max):
        # Validate that the universe has a logical forward progression
        if universe_min >= universe_max:
            raise ValueError("Universe minimum must be strictly less than maximum.")
        
        self.universe_min = float(universe_min)
        self.universe_max = float(universe_max)

    def evaluate(self, x):
        """
        Calculates the membership degree [0, 1] for a given input tensor 'x'.
        Must be explicitly overridden by any mathematical shape inheriting this class.
        """
        raise NotImplementedError("Subclasses must implement the 'evaluate' method.")

    def get_label(self):
        """Generates a dynamic string label for UI/plotting purposes."""
        return "Base Membership Function"

    def plot(self, title="Membership Function"):
        """
        Renders a high-resolution 2D graph of the membership function 
        across the entire initialized universe of discourse.
        """
        # Generate a dense array of 500 points for a smooth, continuous line
        x = np.linspace(self.universe_min, self.universe_max, 500)
        y = self.evaluate(x)
        
        plt.figure(figsize=(8, 5))
        plt.plot(x, y, label=self.get_label(), color='blue', linewidth=2)
        
        # UI formatting for clear analytical reading
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel("Universe of Discourse (x)", fontsize=12)
        plt.ylabel("Degree of Membership A(x)", fontsize=12)
        plt.ylim(-0.05, 1.1) 
        plt.xlim(self.universe_min, self.universe_max)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(loc="upper right")
        plt.show()