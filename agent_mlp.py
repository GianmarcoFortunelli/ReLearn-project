import numpy as np
import torch
import joblib
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import torch.optim as optim
from config import MLP_HIDDEN_LAYERS, MLP_LEARNING_RATE, GAMMA, EPSILON_START, EPSILON_END, EPSILON_DECAY


class MLPNetwork(nn.Module):
    
    def __init__(self, input_dim, output_dim, hidden_layers=MLP_HIDDEN_LAYERS):
        super(MLPNetwork, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class MLPSARSAAgent:

    def __init__(self, state_dim=8, action_dim=3, hidden_layers=MLP_HIDDEN_LAYERS,
                 learning_rate=MLP_LEARNING_RATE, gamma=GAMMA, epsilon_start=EPSILON_START,
                 epsilon_end=EPSILON_END, epsilon_decay=EPSILON_DECAY):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.feature_scaler = None
        # Q-network: takes state and action as input, outputs Q-value
        self.q_network = MLPNetwork(state_dim + action_dim, 1, hidden_layers)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        # policy network: takes state as input, outputs action (portfolio weights)
        self.policy_network = MLPNetwork(state_dim, action_dim, hidden_layers)
        self.policy_optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)

    def _state_to_tensor(self, state):
        features = state['features']
        if self.feature_scaler is not None:
            features = self.feature_scaler.transform(np.array(features).reshape(1, -1))[0]
        state_vector = np.concatenate([features, state['weights']])
        return torch.FloatTensor(state_vector)

    def fit_feature_scaler(self, features):
        self.feature_scaler = StandardScaler()
        self.feature_scaler.fit(features)

    def _get_q_value(self, state, action):
        if isinstance(state, dict):
            state_tensor = self._state_to_tensor(state)
        else:
            state_tensor = state
        
        if isinstance(action, np.ndarray):
            action_tensor = torch.FloatTensor(action)
        else:
            action_tensor = action
        
        state_action = torch.cat([state_tensor, action_tensor])
        return self.q_network(state_action)
    
    def select_action(self, state, greedy=False):
        # Epsilon-greedy exploration
        if not greedy and np.random.random() < self.epsilon:
            # Random action
            action = np.random.dirichlet(np.ones(self.action_dim))
        else:
            # Use policy network
            state_tensor = self._state_to_tensor(state)
            with torch.no_grad():
                action_logits = self.policy_network(state_tensor)
                # Apply softmax to ensure weights sum to 1
                action = torch.softmax(action_logits, dim=0).numpy()
        
        return action
    
    def update(self, state, action, reward, next_state, next_action, done):
        state_tensor = self._state_to_tensor(state)
        action_tensor = torch.FloatTensor(action)
        next_state_tensor = self._state_to_tensor(next_state)
        next_action_tensor = torch.FloatTensor(next_action)
        
        # Current Q-value
        current_q = self._get_q_value(state_tensor, action_tensor)
        
        # Target Q-value
        if done:
            target_q = torch.FloatTensor([reward])
        else:
            with torch.no_grad():
                next_q = self._get_q_value(next_state_tensor, next_action_tensor)
                target_q = reward + self.gamma * next_q
        
        # Update Q-network
        self.optimizer.zero_grad()
        loss = self.criterion(current_q, target_q)
        loss.backward()
        self.optimizer.step()
        
        # Update policy network to maximize Q-value
        self.policy_optimizer.zero_grad()
        state_tensor_grad = self._state_to_tensor(state)
        action_pred = self.policy_network(state_tensor_grad)
        action_pred_softmax = torch.softmax(action_pred, dim=0)
        
        q_value = self._get_q_value(state_tensor_grad, action_pred_softmax)
        policy_loss = -q_value.mean()  # Maximize Q-value
        policy_loss.backward()
        self.policy_optimizer.step()
        
        return loss.item()
    
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def get_action(self, state):
        return self.select_action(state, greedy=True)
    
    def save(self, filepath):
        torch.save({
            'q_network': self.q_network.state_dict(),
            'policy_network': self.policy_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'policy_optimizer': self.policy_optimizer.state_dict()
        }, filepath)
        joblib.dump(self.feature_scaler, filepath + '_scaler.pkl')

    def load(self, filepath):
        checkpoint = torch.load(filepath)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.policy_network.load_state_dict(checkpoint['policy_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer'])
        self.feature_scaler = joblib.load(filepath + '_scaler.pkl')
