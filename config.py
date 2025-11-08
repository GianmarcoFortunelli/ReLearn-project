
# Configuration file for RL Portfolio Management

# Environment parameters
TRANSACTION_COST = 0.0  # 0.25%
INITIAL_PORTFOLIO_VALUE = 1.0
INTEREST_RATE = 0.0  # Daily interest rate for cash

# Data parameters
TRAIN_TEST_SPLIT = 0.8
CV_FOLDS = 5

# Discrete actions for tabular policy
DISCRETE_ACTIONS = [
    [1.0, 0.0, 0.0],    # 100% Stock, 0% Bond, 0% Cash
    [0.6, 0.4, 0.0],    
    [0.5, 0.5, 0.0],   
    [0.4, 0.6, 0.0],    
    [0.0, 1.0, 0.0],    
    [0.33, 0.33, 0.34], 
    [0.0, 0.0, 1.0]    
]

# Tabular policy parameters
NUM_BINS = 3  # k parameter for discretization

# SARSA parameters
ALPHA = 0.1           # Learning rate
GAMMA = 0.99          # Discount factor
EPSILON_START = 1.0   # Initial exploration rate
EPSILON_END = 0.01    # Final exploration rate
EPSILON_DECAY = 0.995 # Epsilon decay rate

# Training parameters
NUM_EPISODES = 500

# Features (excluding prices which are only for evaluation)
FEATURE_COLUMNS = ['stock_momentum', 'stock_volatility', 'bond_momentum', 'bond_volatility', 'correlation']
PRICE_COLUMNS = ['stock_price', 'bond_price']

# Random seed for reproducibility
RANDOM_SEED = 42
