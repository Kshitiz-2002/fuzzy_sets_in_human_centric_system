import numpy as np

class FuzzySetAnalysis:
    """
    Evaluates and measures the properties of a fuzzy set across a discrete universe of discourse.
    Accepts any membership function class (e.g., TriangularMembershipFunction) 
    that contains an 'evaluate(x)' method.
    """
    def __init__(self, universe, membership_function):
        self.universe = np.asarray(universe)
        self.mem_func = membership_function
        self.degrees = self.mem_func.evaluate(self.universe)

    def height(self):
        """
        Height of a fuzzy set: The absolute maximum membership degree present in the set.
        Returns a scalar value between 0 and 1.
        """
        return np.max(self.degrees)

    def normalize(self):
        """
        Normalization Operation: Scales all membership degrees proportionally so the 
        set's peak height reaches exactly 1.0. Returns a new array of scaled degrees.
        """
        h = self.height()
        if h == 0:
            return self.degrees
        return self.degrees / h

    def support(self):
        """
        Support of a fuzzy set: Returns all elements in the universe with a 
        membership degree strictly greater than 0 (total exclusions are filtered out).
        """
        return self.universe[self.degrees > 0]

    def core(self):
        """
        Core of a fuzzy set: Returns all elements in the universe possessing 
        absolute, full membership (degree exactly equal to 1.0).
        """
        return self.universe[self.degrees == 1.0]

    def is_equal(self, other_fuzzy_set):
        """
        Equality: Two fuzzy sets are strictly equal if their membership degrees 
        are identical for every single element across the universe of discourse.
        """
        return np.allclose(self.degrees, other_fuzzy_set.degrees)

    def is_included_in(self, other_fuzzy_set):
        """
        Inclusion (Subset): Fuzzy set A is included in B if A's membership degree 
        is less than or equal to B's membership degree for every element.
        """
        return np.all(self.degrees <= other_fuzzy_set.degrees)

    def cardinality(self):
        """
        Cardinality (Sigma-count): Measures information granularity by summing 
        all membership degrees. A higher cardinality indicates a "larger" or 
        more encompassing fuzzy set.
        """
        return np.sum(self.degrees)

    def energy(self):
        """
        Energy Measure: Quantifies the total activation or "mass" of the fuzzy set. 
        In discrete finite spaces, this is mathematically identical to scalar cardinality.
        """
        return self.cardinality()

    def entropy(self):
        """
        Entropy Measure (De Luca & Termini): Quantifies the level of fuzziness or 
        ambiguity. Entropy is 0 for crisp sets (only 0s and 1s) and reaches maximum 
        when elements have a membership of 0.5 (maximum ambiguity).
        """
        # Filter out 0s and 1s to avoid undefined log(0) calculations
        ambiguous = self.degrees[(self.degrees > 0) & (self.degrees < 1)]
        if len(ambiguous) == 0:
            return 0.0
        
        # Shannon-like entropy calculation for fuzzy logic
        ent = -np.sum(ambiguous * np.log(ambiguous) + (1 - ambiguous) * np.log(1 - ambiguous))
        return ent

    def alpha_cut(self, alpha):
        """
        Alpha-cut: Returns the crisp set of elements in the universe whose 
        membership degree meets or exceeds the specified threshold 'alpha'.
        """
        if not (0 <= alpha <= 1):
            raise ValueError("Alpha must be strictly between 0 and 1.")
        return self.universe[self.degrees >= alpha]

    def representation_theorem(self):
        """
        Representation Theorem: Demonstrates that any fuzzy set can be perfectly 
        reconstructed by taking the supremum (maximum) of its scaled alpha-cuts.
        Returns an array identical to the original 'self.degrees'.
        """
        unique_alphas = np.unique(self.degrees)
        reconstructed_degrees = np.zeros_like(self.degrees)
        
        for alpha in unique_alphas:
            # Create a binary mask of the alpha-cut (1 if in the cut, 0 otherwise)
            crisp_cut_mask = (self.degrees >= alpha).astype(float)
            
            # Scale the crisp mask by alpha and update the supremum
            scaled_cut = alpha * crisp_cut_mask
            reconstructed_degrees = np.maximum(reconstructed_degrees, scaled_cut)
            
        return reconstructed_degrees

    def specificity_interval(self, interval_start, interval_end):
        """
        Specificity of Sets (Crisp Intervals): Measures how precise or narrow an interval is 
        relative to the entire universe. 1 means perfectly specific (a single point), 
        while 0 means completely unspecific (spans the entire universe).
        """
        universe_range = self.universe[-1] - self.universe[0]
        if universe_range == 0:
            return 1.0
        
        interval_length = interval_end - interval_start
        return 1.0 - (interval_length / universe_range)

    def specificity_alpha_cut(self, alpha):
        """
        Specificity of Alpha-cuts: Extracts the crisp boundary of an alpha-cut 
        and calculates its specificity as a standard interval.
        """
        cut_elements = self.alpha_cut(alpha)
        if len(cut_elements) == 0:
            return 0.0
        
        # Calculate the interval from the lowest to highest element in the cut
        return self.specificity_interval(cut_elements[0], cut_elements[-1])

    def specificity_fuzzy_set(self):
        """
        Specificity of Fuzzy Sets (Yager's Measure): Evaluates how concentrated the 
        fuzzy set is around its core. Calculated by integrating (summing in discrete form) 
        the inverse cardinality of its alpha-cuts over all membership levels.
        """
        alphas = np.sort(np.unique(self.degrees))
        if alphas[0] != 0:
            alphas = np.insert(alphas, 0, 0.0)
            
        specificity = 0.0
        
        for i in range(1, len(alphas)):
            delta_alpha = alphas[i] - alphas[i-1]
            cut_elements = self.alpha_cut(alphas[i])
            cardinality_of_cut = len(cut_elements)
            
            if cardinality_of_cut > 0:
                specificity += delta_alpha / cardinality_of_cut
                
        return specificity


from mem_functions.triangularMemFunc import TriangularMembershipFunction
if __name__ == "__main__":
    
    # 1. Define the discrete universe of discourse (e.g., temperatures 0 to 40)
    X = np.linspace(0, 40, 100)
    
    # 2. Mocking the initialization (requires your actual imported class)
    mem_func = TriangularMembershipFunction(a=15, m=25, b=35, universe_min=0, universe_max=40)
    
    # 3. Analyze the set
    analysis = FuzzySetAnalysis(universe=X, membership_function=mem_func)
    print(f"Height: {analysis.height()}")
    print(f"Entropy: {analysis.entropy():.4f}")
    print(f"Specificity: {analysis.specificity_fuzzy_set():.4f}")