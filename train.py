import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from environment import PortfolioEnv
from agent_tabular import TabularSARSAAgent
from agent_mlp import MLPSARSAAgent
from performance_evaluation import evaluation
from config import (NUM_EPISODES, TRAIN_TEST_SPLIT, NUM_BINS,
                    FEATURE_COLUMNS, RANDOM_SEED, UPDATE_RATE)


def train_episode(env, agent, policy_type='tabular'):
    
    state = env.reset()
    
    # Both agents now return (action_idx, action)
    action_idx, action = agent.select_action(state)
    
    total_reward = 0
    episode_length = 0
    losses = []
    
    while True:
        next_state, reward, done, _ = env.step(action)
        
        # Both agents now return (action_idx, action)
        next_action_idx, next_action = agent.select_action(next_state)
        
        # Update agent (stores transition for MLP, updates Q-table for tabular)
        agent.update(state, action_idx, reward, next_state, next_action_idx, done)
        
        # For MLP: perform mini-batch learning every few steps
        if policy_type == 'mlp' and episode_length % UPDATE_RATE == 0:
            loss = agent.update_network(update_rate=1)
            if loss > 0:
                losses.append(loss)
        
        state = next_state
        action_idx = next_action_idx
        action = next_action
        total_reward += reward
        episode_length += 1
        
        if done:
            break
    
    # Final network update for MLP at end of episode
    if policy_type == 'mlp':
        loss = agent.update_network(update_rate=UPDATE_RATE)
        if loss > 0:
            losses.append(loss)
    
    avg_loss = np.mean(losses) if losses else 0
    return total_reward, episode_length, avg_loss


def train_agent(train_data, test_data, policy_type='tabular', n_episodes=NUM_EPISODES, 
                n_bins=NUM_BINS):
    np.random.seed(RANDOM_SEED)
    
    # Initialize agent
    if policy_type == 'tabular':
        agent = TabularSARSAAgent(n_bins=n_bins)
        # Fit discretizer on training data
        features = train_data[FEATURE_COLUMNS].values
        agent.fit_discretizer(features)
    else:  # mlp
        agent = MLPSARSAAgent(state_dim=8)
        features = train_data[FEATURE_COLUMNS].values
        agent.fit_feature_scaler(features)
    
    # Initialize environment
    env = PortfolioEnv(train_data)
    
    # Training statistics
    episode_rewards = []
    episode_lengths = []
    episode_returns = []
    episode_losses = []
    epsilons = []
    
    # Training loop
    for episode in range(n_episodes):
        total_reward, length, avg_loss = train_episode(env, agent, policy_type)
        annual_return = env.get_annual_return()
        
        episode_rewards.append(total_reward)
        episode_lengths.append(length)
        episode_returns.append(annual_return)
        episode_losses.append(avg_loss)
        epsilons.append(agent.epsilon)
        
        # Decay epsilon
        agent.decay_epsilon()
        
        # Print statistics 
        if (episode + 1) % 20 == 0:
            avg_reward = np.mean(episode_rewards[-20:])
            avg_return = np.mean(episode_returns[-20:])
            print(f"Episode {episode + 1}/{n_episodes} - "
                  f"Avg Reward: {avg_reward:.4f}, "
                  f"Avg Return: {avg_return:.4f}, "
                  f"Epsilon: {agent.epsilon:.4f}")
            if policy_type == 'mlp' and len(episode_losses) > 0:
                avg_loss = np.mean([l for l in episode_losses[-20:] if l > 0])
                if not np.isnan(avg_loss):
                    print(f"  Avg Loss: {avg_loss:.6f}")
            evaluation(agent, test_data, plot=False)
    
    training_stats = {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'episode_returns': episode_returns,
        'episode_losses': episode_losses,
        'epsilons': epsilons
    }
    
    return agent, training_stats


def plot_training_stats(stats, save_path=None):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot rewards
    axes[0, 0].plot(stats['episode_rewards'])
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Annual Reward')
    axes[0, 0].grid(True)
    
    # Plot returns
    axes[0, 1].plot(stats['episode_returns'])
    axes[0, 1].set_title('Episode Returns')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Annual Return (%)')
    axes[0, 1].grid(True)
    
    # Plot epsilon
    axes[1, 0].plot(stats['epsilons'])
    axes[1, 0].set_title('Exploration Rate (Epsilon)')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Epsilon')
    axes[1, 0].grid(True)
    
    # Plot losses (for MLP)
    if stats['episode_losses'] and any(l > 0 for l in stats['episode_losses']):
        axes[1, 1].plot(stats['episode_losses'][20:])
        axes[1, 1].set_title('Episode Losses')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Average Loss')
        axes[1, 1].grid(True)
    else:
        axes[1, 1].text(0.5, 0.5, 'Loss not applicable\n(Tabular Policy)', 
                       ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Episode Losses')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training plots saved to {save_path}")
    
    plt.show()


if __name__ == "__main__":
    # Load data
    print("Loading data...")
    data = pd.read_csv('dataset.csv')
    
    # Split into train and test
    split_idx = int(len(data) * TRAIN_TEST_SPLIT)
    train_data = data.iloc[:split_idx].reset_index(drop=True)
    test_data = data.iloc[split_idx:].reset_index(drop=True)
    
    print(f"Train size: {len(train_data)}, Test size: {len(test_data)}")
    
    # Choose policy type
    policy_type = 'mlp'  # 'tabular' or 'mlp'
    print(f"\nTraining {policy_type.upper()} policy...")
    
    # Train agent
    agent, training_stats = train_agent(train_data, test_data, policy_type=policy_type)
    
    # Plot training statistics
    plot_training_stats(training_stats, save_path=f'training_stats_{policy_type}.png')
    
    # Save agent
    if policy_type == 'tabular':
        agent.save(f'agent_{policy_type}.npy')
    else:
        agent.save(f'agent_{policy_type}.pth')
    
    print(f"\nAgent saved to agent_{policy_type}.{'npy' if policy_type == 'tabular' else 'pth'}")
