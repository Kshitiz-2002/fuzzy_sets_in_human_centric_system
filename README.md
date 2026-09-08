# Fuzzy Set Analysis & Neuro-Fuzzy Library

A Python library for modeling, evaluating, and analyzing fuzzy sets and membership functions across discrete and continuous universes of discourse. This library provides foundational mathematical structures for cognitive modeling, human-centric systems, and fuzzy logic controllers.

## Features

* **Membership Functions:** Includes robust, vectorized implementations of common fuzzy shapes:
  * Triangular
  * Trapezoidal
  * Gaussian
  * Gamma
  * S-Shaped
  * Exponential
  * Custom Piecewise
* **Fuzzy Set Analysis:** A comprehensive analysis suite (`FuzzySetAnalysis`) to calculate:
  * Height, Support, and Core
  * Normalization
  * Cardinality (Sigma-count) and Energy
  * Entropy (De Luca & Termini measure)
  * Alpha-cuts and Representation Theorem reconstruction
  * Specificity (Yager's measure)
* **Visualization:** Built-in `matplotlib` integration for high-resolution plotting of any defined fuzzy set.

## Quick Start

```python
import numpy as np
from mem_functions.triangle import TriangularMembershipFunction
from fuzzy_sets.main import FuzzySetAnalysis

# 1. Define the universe of discourse (e.g., Temperature from 0 to 40)
X = np.linspace(0, 40, 100)

# 2. Initialize a membership function
optimal_temp = TriangularMembershipFunction(a=15, m=25, b=35, universe_min=0, universe_max=40)

# 3. Analyze the fuzzy set
analysis = FuzzySetAnalysis(universe=X, membership_function=optimal_temp)

print(f"Height: {analysis.height()}")
print(f"Entropy: {analysis.entropy():.4f}")
print(f"Specificity: {analysis.specificity_fuzzy_set():.4f}")

# 4. Plot the function
optimal_temp.plot(title="Target Room Temperature")