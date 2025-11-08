
# Configuration file for RL Portfolio Management

# Environment parameters
TRANSACTION_COST = 0.0
INITIAL_PORTFOLIO_VALUE = 1.0

# Data parameters
TRAIN_TEST_SPLIT = 0.7

# Discrete actions for tabular policy
DISCRETE_ACTIONS = [
    [1.0, 0.0, 0.0],    # 100% Stock, 0% Bond, 0% Cash
    [0.8, 0.2, 0.0],
    [0.6, 0.4, 0.0],
    [0.6, 0.3, 0.1],
    [0.7, 0.2, 0.1],
    [0.7, 0.1, 0.2],    
    [0.5, 0.5, 0.0],   
    [0.4, 0.6, 0.0], 
    [0.2, 0.8, 0.0],   
    [0.0, 1.0, 0.0],    
    [0.33, 0.33, 0.34], 
    [0.0, 0.0, 1.0]    
]

# Tabular policy parameters
NUM_BINS = 7  # k parameter for discretization

# SARSA parameters
ALPHA = 0.1           # Learning rate
GAMMA = 0.99          # Discount factor

EPSILON_START = 1.0   # Initial exploration rate
EPSILON_END = 0.01    # Final exploration rate
EPSILON_DECAY = 0.99 # Epsilon decay rate

# Training parameters
NUM_EPISODES = 200

# Features (excluding prices which are only for evaluation)
FEATURE_COLUMNS = ['stock_momentum', 'stock_volatility', 'bond_momentum', 'bond_volatility', 'correlation']
PRICE_COLUMNS = ['stock_price', 'bond_price']

# Random seed for reproducibility
RANDOM_SEED = 42

# MLP parameters
MLP_HIDDEN_LAYERS = [32, 128, 128, 128]  # Standard architecture
MLP_LEARNING_RATE = 0.0001

# MLP Discrete actions (more granular than tabular)
MLP_DISCRETE_ACTIONS = [
    [1.0, 0.0, 0.0],    # 100% Stock
    [0.9, 0.1, 0.0],
    [0.9, 0.0, 0.1],
    [0.8, 0.2, 0.0],
    [0.8, 0.1, 0.1],
    [0.8, 0.0, 0.2],
    [0.7, 0.3, 0.0],
    [0.7, 0.2, 0.1],
    [0.7, 0.1, 0.2],
    [0.7, 0.0, 0.3],
    [0.6, 0.4, 0.0],
    [0.6, 0.3, 0.1],
    [0.6, 0.2, 0.2],
    [0.6, 0.1, 0.3],
    [0.6, 0.0, 0.4],
    [0.5, 0.5, 0.0],
    [0.5, 0.4, 0.1],
    [0.5, 0.3, 0.2],
    [0.5, 0.2, 0.3],
    [0.5, 0.1, 0.4],
    [0.5, 0.0, 0.5],
    [0.4, 0.6, 0.0],
    [0.4, 0.5, 0.1],
    [0.4, 0.4, 0.2],
    [0.4, 0.3, 0.3],
    [0.4, 0.2, 0.4],
    [0.4, 0.1, 0.5],
    [0.4, 0.0, 0.6],
    [0.3, 0.7, 0.0],
    [0.3, 0.6, 0.1],
    [0.3, 0.5, 0.2],
    [0.3, 0.4, 0.3],
    [0.3, 0.3, 0.4],
    [0.3, 0.2, 0.5],
    [0.3, 0.1, 0.6],
    [0.3, 0.0, 0.7],
    [0.2, 0.8, 0.0],
    [0.2, 0.7, 0.1],
    [0.2, 0.6, 0.2],
    [0.2, 0.5, 0.3],
    [0.2, 0.4, 0.4],
    [0.2, 0.3, 0.5],
    [0.2, 0.2, 0.6],
    [0.2, 0.1, 0.7],
    [0.2, 0.0, 0.8],
    [0.1, 0.9, 0.0],
    [0.1, 0.8, 0.1],
    [0.1, 0.7, 0.2],
    [0.1, 0.6, 0.3],
    [0.1, 0.5, 0.4],
    [0.1, 0.4, 0.5],
    [0.1, 0.3, 0.6],
    [0.1, 0.2, 0.7],
    [0.1, 0.1, 0.8],
    [0.1, 0.0, 0.9],
    [0.0, 1.0, 0.0],    # 100% Bond
    [0.0, 0.9, 0.1],
    [0.0, 0.8, 0.2],
    [0.0, 0.7, 0.3],
    [0.0, 0.6, 0.4],
    [0.0, 0.5, 0.5],
    [0.0, 0.4, 0.6],
    [0.0, 0.3, 0.7],
    [0.0, 0.2, 0.8],
    [0.0, 0.1, 0.9],
    [0.0, 0.0, 1.0],    # 100% Cash
    [0.33, 0.33, 0.34], # Equal weight
]

# Replay buffer parameters
REPLAY_BUFFER_SIZE = 5000
BATCH_SIZE = 32
UPDATE_RATE = 4  # Update network every N steps
