"""
Utility functions for state discretization
"""
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer


class StateDiscretizer:    
    def __init__(self, n_bins=3):
        self.n_bins = n_bins
        self.discretizer = None
        self.n_features = None
        
    def fit(self, features):
        self.n_features = features.shape[1]
        self.discretizer = KBinsDiscretizer(
            n_bins=self.n_bins,
            encode='ordinal',
            strategy='quantile'
        )
        self.discretizer.fit(features)
        
    def transform(self, features):
        discretized = self.discretizer.transform(features).astype(int)
        
        # Convert multi-dimensional discrete state to single index
        state_indices = []
        for row in discretized:
            state_idx = 0
            for i, val in enumerate(row):
                state_idx += val * (self.n_bins ** i)
            state_indices.append(state_idx)
        
        return state_indices[0] if len(state_indices) == 1 else state_indices
    
    def get_n_states(self):
        return self.n_bins ** self.n_features


def discretize_weights(weights, n_bins=3):
    discretized = np.floor(weights * n_bins).astype(int)
    discretized = np.clip(discretized, 0, n_bins - 1)
    
    # Convert to single index
    idx = discretized[0] * (n_bins ** 2) + discretized[1] * n_bins + discretized[2]
    return idx
