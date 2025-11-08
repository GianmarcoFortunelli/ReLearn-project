import numpy as np
import torch
import joblib
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import torch.optim as optim
from config import (MLP_HIDDEN_LAYERS, MLP_LEARNING_RATE, GAMMA, EPSILON_START, 
                    EPSILON_END, EPSILON_DECAY, MLP_DISCRETE_ACTIONS, 
                    REPLAY_BUFFER_SIZE, BATCH_SIZE)


class QNetwork(nn.Module):
    
    def __init__(self, state_dim, action_dim, hidden_layers=MLP_HIDDEN_LAYERS):
        super(QNetwork, self).__init__()
        
        layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class ReplayBuffer:
    
    def __init__(self, max_size=REPLAY_BUFFER_SIZE):
        self.buffer = []
        self.max_size = max_size
        
    def add_to_buffer(self, transition):
        if len(self.buffer) >= self.max_size:
            self.buffer.pop(0)
        self.buffer.append(transition)
    
    def sample_minibatch(self, batch_size):
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
        
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        
        states = []
        next_states = []
        actions = []
        next_actions = []
        rewards = []
        terminals = []
        
        for idx in indices:
            transition = self.buffer[idx]
            states.append(transition[0])
            next_states.append(transition[1])
            actions.append(transition[2])
            next_actions.append(transition[3])
            rewards.append(transition[4])
            terminals.append(transition[5])
        
        # Convert to numpy arrays first, then to tensors (more efficient)
        return (torch.from_numpy(np.array(states)).float(), 
                torch.from_numpy(np.array(next_states)).float(), 
                torch.from_numpy(np.array(actions)).float(), 
                torch.from_numpy(np.array(next_actions)).float(),
                torch.from_numpy(np.array(rewards)).float(), 
                torch.from_numpy(np.array(terminals)).float())
    
    def __len__(self):
        return len(self.buffer)


class MLPSARSAAgent:
    def __init__(self, state_dim=8, hidden_layers=MLP_HIDDEN_LAYERS,
                 learning_rate=MLP_LEARNING_RATE, gamma=GAMMA, epsilon_start=EPSILON_START,
                 epsilon_end=EPSILON_END, epsilon_decay=EPSILON_DECAY):
        self.state_dim = state_dim
        self.actions = MLP_DISCRETE_ACTIONS
        self.n_actions = len(self.actions)
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.feature_scaler = None
        
        # Q-network: takes state as input, outputs Q-value for each action
        self.q_network = QNetwork(state_dim, self.n_actions, hidden_layers)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer()

    def _state_to_tensor(self, state):
        features = state['features']
        if self.feature_scaler is not None:
            features = self.feature_scaler.transform(np.array(features).reshape(1, -1))[0]
        state_vector = np.concatenate([features, state['weights']])
        return torch.FloatTensor(state_vector)

    def fit_feature_scaler(self, features):
        self.feature_scaler = StandardScaler()
        self.feature_scaler.fit(features)
    
    def select_action(self, state, greedy=False):
        if not greedy and np.random.random() < self.epsilon:
            # Random action
            action_idx = np.random.randint(self.n_actions)
        else:
            # Greedy action: select action with highest Q-value
            state_tensor = self._state_to_tensor(state)
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
                action_idx = torch.argmax(q_values).item()
        
        action = np.array(self.actions[action_idx])
        return action_idx, action
    
    def update(self, state, action_idx, reward, next_state, next_action_idx, done):
        state_vector = self._state_to_tensor(state).numpy()
        next_state_vector = self._state_to_tensor(next_state).numpy()
        
        self.replay_buffer.add_to_buffer(
            (state_vector, next_state_vector, [action_idx], [next_action_idx], [reward], [int(done)])
        )
    
    def update_network(self, update_rate):
        if len(self.replay_buffer) < BATCH_SIZE:
            return 0.0
        
        total_loss = 0.0
        for _ in range(update_rate):
            states, next_states, actions, next_actions, rewards, terminals = \
                self.replay_buffer.sample_minibatch(BATCH_SIZE)
            
            # Current Q-values
            q_values = self.q_network(states)
            current_q = torch.gather(q_values, dim=1, index=actions.long())
            
            # Target Q-values (SARSA: use next action's Q-value)
            with torch.no_grad():
                next_q_values = self.q_network(next_states)
                next_q = torch.gather(next_q_values, dim=1, index=next_actions.long())
                
                not_terminals = 1 - terminals
                target_q = rewards + not_terminals * self.gamma * next_q
            
            # Compute loss and update
            loss = self.criterion(current_q, target_q)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / update_rate
    
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def get_action(self, state):
        _, action = self.select_action(state, greedy=True)
        return action
    
    def save(self, filepath):
        torch.save({
            'q_network': self.q_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, filepath)
        joblib.dump(self.feature_scaler, filepath + '_scaler.pkl')

    def load(self, filepath):
        checkpoint = torch.load(filepath)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.feature_scaler = joblib.load(filepath + '_scaler.pkl')
