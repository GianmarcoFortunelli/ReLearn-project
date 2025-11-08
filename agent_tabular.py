import numpy as np
from config import DISCRETE_ACTIONS, ALPHA, GAMMA, EPSILON_START, EPSILON_END, EPSILON_DECAY
from utils import StateDiscretizer, discretize_weights


class TabularSARSAAgent:
    def __init__(self, n_bins=3, alpha=ALPHA, gamma=GAMMA, 
                 epsilon_start=EPSILON_START, epsilon_end=EPSILON_END, 
                 epsilon_decay=EPSILON_DECAY):
        self.n_bins = n_bins
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        self.actions = DISCRETE_ACTIONS
        self.n_actions = len(self.actions)
        
        self.discretizer = StateDiscretizer(n_bins=n_bins)
        self.q_table = {}  # Dictionary for Q-values: (state, action) -> value
        
        self.is_fitted = False
        
    def fit_discretizer(self, features):
        self.discretizer.fit(features)
        self.is_fitted = True
        
    def _get_state_key(self, state):
        feature_idx = self.discretizer.transform(state['features'])
        weight_idx = discretize_weights(state['weights'], self.n_bins)
        return (feature_idx, weight_idx)
    
    def _get_q_value(self, state_key, action_idx):
        return self.q_table.get((state_key, action_idx), 0.0)
    
    def _set_q_value(self, state_key, action_idx, value):
        self.q_table[(state_key, action_idx)] = value
    
    def select_action(self, state, greedy=False):
        if not self.is_fitted:
            print("ERROR: Discretizer not fitted. Call fit_discretizer() first.")
            return 
        
        state_key = self._get_state_key(state)
        
        # Epsilon-greedy action selection
        if not greedy and np.random.random() < self.epsilon:
            action_idx = np.random.randint(self.n_actions)
        else:
            # Select action with highest Q-value
            q_values = [self._get_q_value(state_key, a) for a in range(self.n_actions)]
            action_idx = np.argmax(q_values)
        
        action = np.array(self.actions[action_idx])
        return action_idx, action
    
    def update(self, state, action_idx, reward, next_state, next_action_idx, done):
        state_key = self._get_state_key(state)
        next_state_key = self._get_state_key(next_state)
        
        current_q = self._get_q_value(state_key, action_idx)
        
        if done:
            target = reward
        else:
            next_q = self._get_q_value(next_state_key, next_action_idx)
            target = reward + self.gamma * next_q
        
        # SARSA update
        new_q = current_q + self.alpha * (target - current_q)
        self._set_q_value(state_key, action_idx, new_q)
    
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def get_action(self, state):
        _, action = self.select_action(state, greedy=True)
        return action
    
    def save(self, filepath):
        np.save(filepath, {
            'q_table': self.q_table,
            'n_bins': self.n_bins,
            'discretizer': self.discretizer
        })
    
    def load(self, filepath):
        data = np.load(filepath, allow_pickle=True).item()
        self.q_table = data['q_table']
        self.n_bins = data['n_bins']
        self.discretizer = data['discretizer']
        self.is_fitted = True
