import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from environment import PortfolioEnv

def evaluation(agent, test_set):
    static_weights = [0.6, 0.4, 0.0]
    env_model = PortfolioEnv(test_set)
    env_static = PortfolioEnv(test_set)

    n = test_set.shape[1]
    total_value_model = [0.0] * n
    total_value_static = [0.0] * n

    actions = []

    total_value_model = [1.0]
    total_value_static = [1.0]

    state = env_model.reset()

    state = env_model.reset()
    while True:
        action = agent.get_action(state)
        actions.append[action]
        next_state, reward, done, info = env_model.step(action)
        if done:
            break
        total_value_model.append(info['portfolio_value'])
        state = next_state
    
    state = env_static.reset()
    
    while True:
        next_state, reward, done, info = env_static.step(static_weights)
        if done:
            break
        total_value_static.append(info['portfolio_value'])
        state = next_state

    static_overall_gain = (total_value_static[-1] - 1) * 100
    model_overall_gain = (total_value_model[-1] - 1) * 100

    # print("static overall gain: ", static_overall_gain)
    # print("model overall gain: ", model_overall_gain)

    returns_static = np.diff(total_value_static) / np.array(total_value_static[:-1])
    returns_model = np.diff(total_value_model) / np.array(total_value_model[:-1])

    sharpe_static = np.mean(returns_static) / np.std(returns_static, ddof=1) * np.sqrt(252)
    sharpe_model = np.mean(returns_model) / np.std(returns_model, ddof=1) * np.sqrt(252)
    
    # print("static sharpe ratio: ", sharpe_static)
    # print("model sharpe ratio: ", sharpe_model)

    def max_drawdown(values):
        peak = np.maximum.accumulate(values)
        drawdown = (values - peak) / peak
        return np.min(drawdown) * 100
    
    mdd_model = max_drawdown(total_value_model)
    mdd_static = max_drawdown(total_value_static)
    
    # print("static maximum drowdown: ", mdd_static)
    # print("model smaximum drowdown: ", mdd_model)

    vol_model = np.std(returns_model, ddof=1) * np.sqrt(252) * 100
    vol_static = np.std(returns_static, ddof=1) * np.sqrt(252) * 100

    # print("volatility static: ", vol_static)
    # print("volatility model: ", vol_model)

    print("\n" + "="*60)
    print("PERFORMANCE EVALUATION RESULTS")
    print("="*60)
    
    print("\nRL Model:")
    print(f"  Total Return:        {model_overall_gain:>8.2f}%")
    print(f"  Sharpe Ratio:        {sharpe_model:>8.2f}")
    print(f"  Volatility (Annual): {vol_model:>8.2f}%")
    print(f"  Max Drawdown:        {mdd_model:>8.2f}%")
    
    print("\n60/40 Benchmark:")
    print(f"  Total Return:        {static_overall_gain:>8.2f}%")
    print(f"  Sharpe Ratio:        {sharpe_static:>8.2f}")
    print(f"  Volatility (Annual): {vol_static:>8.2f}%")
    print(f"  Max Drawdown:        {mdd_static:>8.2f}%")
    
    print("\nOutperformance:")
    print(f"  Return Difference:   {model_overall_gain - static_overall_gain:>8.2f}%")
    print(f"  Sharpe Difference:   {sharpe_model - sharpe_static:>8.2f}")
    
    print("="*60 + "\n")

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    dates = pd.to_datetime(test_set['date'].values)
    plt.title("Portfolio manager")
    axes[0, 0].plot(dates, actions[0], linewidth=2, label='Stock')
    axes[0, 0].plot(dates, actions[1], linewidth=2, label='Bonds')
    axes[0, 0].plot(dates, actions[2], linewidth=2, label='Cash')
    axes[0, 0].set_title('Weights')
    axes[0, 0].set_xlabel('Date')
    axes[0, 0].set_ylabel('Portfolio distributions')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    cum_returns_model = (np.array(total_value_model) - 1) * 100
    cum_returns_benchmark = (np.array(total_value_static) - 1) * 100
    axes[0, 1].plot(dates, cum_returns_model, linewidth=2, label='RL Model')
    axes[0, 1].plot(dates, cum_returns_benchmark, linewidth=2, label='60/40 Benchmark', alpha=0.7)
    axes[0, 1].set_title('Cumulative Returns')
    axes[0, 1].set_xlabel('Date')
    axes[0, 1].set_ylabel('Return (%)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    def calculate_drawdown(values):
        peak = np.maximum.accumulate(values)
        drawdown = (values - peak) / peak * 100
        return drawdown
    
    dd_model = calculate_drawdown(total_value_model)
    dd_benchmark = calculate_drawdown(total_value_static)
    axes[1, 0].fill_between(dates, dd_model, 0, alpha=0.5, label='RL Model')
    axes[1, 0].fill_between(dates, dd_benchmark, 0, alpha=0.5, label='60/40 Benchmark')
    axes[1, 0].set_title('Drawdown')
    axes[1, 0].set_xlabel('Date')
    axes[1, 0].set_ylabel('Drawdown (%)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    metrics_text = f"""
    Performance Metrics:
    
    {'Metric':<20} {'RL Model':<15} {'Benchmark':<15}
    {'-'*50}
    {'Total Return (%)':<20} {model_overall_gain:>14.2f} {static_overall_gain:>14.2f}
    {'Sharpe Ratio':<20} {sharpe_model:>14.2f} {sharpe_static:>14.2f}
    {'Volatility (%)':<20} {vol_model:>14.2f} {vol_static:>14.2f}
    {'Max Drawdown (%)':<20} {mdd_model:>14.2f} {mdd_static:>14.2f}
    """
    
    axes[1, 1].text(0.1, 0.5, metrics_text, transform=axes[1, 1].transAxes,
                   fontfamily='monospace', fontsize=10, verticalalignment='center')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig("evaluation_plot.png")
    plt.show()

    return sharpe_static, sharpe_model


if __name__ == "__main__":
    from agent_tabular import TabularSARSAAgent
    from config import TRAIN_TEST_SPLIT
    
    # Load data
    print("Loading data...")
    data = pd.read_csv('dataset.csv')
    
    # Split into train and test
    split_idx = int(len(data) * TRAIN_TEST_SPLIT)
    test_data = data.iloc[split_idx:].reset_index(drop=True)
    
    print(f"Test data size: {len(test_data)}")
    
    # Load trained agent
    agent = TabularSARSAAgent()
    agent.load(f'agent.npy')
    
    # Evaluate agent
    print("Evaluating agent on test data...")
    results = evaluation(agent, test_data)
    