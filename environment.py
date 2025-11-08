import numpy as np
from config import TRANSACTION_COST, INITIAL_PORTFOLIO_VALUE

class PortfolioEnv:
    """
    Environment for portfolio management with 3 assets: stock, bond, cash
    """
    
    def __init__(self, data_df, transaction_cost=TRANSACTION_COST):
        self.data = data_df.reset_index(drop=True)
        self.transaction_cost = transaction_cost
        self.n_steps = len(self.data)
        
        # Portfolio state
        self.current_step = 0
        self.portfolio_value = INITIAL_PORTFOLIO_VALUE
        self.weights = np.array([0.0, 0.0, 1.0])  # Start with 100% cash
        self.done = False
        
        # For reward calculation
        self.initial_value = INITIAL_PORTFOLIO_VALUE
        
    def reset(self):
        self.current_step = 0
        self.portfolio_value = INITIAL_PORTFOLIO_VALUE
        self.weights = np.array([0.0, 0.0, 1.0])  # Start with 100% cash
        self.done = False
        self.initial_value = INITIAL_PORTFOLIO_VALUE
        return self._get_state()
    
    def _get_state(self):
        if self.current_step >= self.n_steps:
            # Return last valid state
            row = self.data.iloc[-1]
        else:
            row = self.data.iloc[self.current_step]
        
        features = np.array([
            row['stock_momentum'],
            row['stock_volatility'],
            row['bond_momentum'],
            row['bond_volatility'],
            row['correlation']
        ])
        
        return {
            'features': features,
            'weights': self.weights.copy(),
            'prices': np.array([row['stock_price'], row['bond_price']])
        }
    
    def step(self, action):
        if self.done:
            print("ERROR no more step left")
            return self._get_state(), None, self.done, None
        
        action = np.array(action)
        if np.sum(action) != 1:
            print("WARN action does not sum to 1, rebalancing")
            action = np.clip(action, 0, 1)
            action = action / np.sum(action)
        
        # Calculate transaction costs
        weight_change = np.abs(action - self.weights)
        transaction_cost = self.transaction_cost * np.sum(weight_change) * self.portfolio_value
        
        current_prices = np.array([
            self.data.iloc[self.current_step]['stock_price'],
            self.data.iloc[self.current_step]['bond_price']
        ])
        
        self.current_step += 1
        
        if self.current_step >= self.n_steps:
            self.done = True
            # For final step, use same prices (no return)
            next_prices = current_prices
        else:
            next_prices = np.array([
                self.data.iloc[self.current_step]['stock_price'],
                self.data.iloc[self.current_step]['bond_price']
            ])
        
        # Calculate returns for each asset
        stock_return = (next_prices[0] / current_prices[0]) - 1
        bond_return = (next_prices[1] / current_prices[1]) - 1
        cash_return = 0.0  # Cash has no return
        
        returns = np.array([stock_return, bond_return, cash_return])
        
        # Calculate portfolio return
        portfolio_return = np.dot(action, returns)
        
        # Update portfolio value after transaction costs
        old_value = self.portfolio_value
        self.portfolio_value = self.portfolio_value * (1 + portfolio_return) - transaction_cost
        
        self.weights = action
        
        # Calculate reward as logarithmic return
        if self.portfolio_value == 0:
            print("WARN portfolio value = 0")
            self.done = True
        
        # Daily log return
        reward = np.log(self.portfolio_value / old_value)

        
        next_state = self._get_state()
        
        info = {
            'portfolio_value': self.portfolio_value,
            'returns': returns,
            'step': self.current_step
        }
        
        return next_state, reward, self.done, info
    
    def get_cumulative_return(self):
        return (self.portfolio_value - self.initial_value) / self.initial_value
