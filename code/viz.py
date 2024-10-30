import matplotlib.pyplot as plt
import numpy as np

# Back-testing results (replace these with your actual values from back-testing)
rl_total_reward = -0.00014039198867976665
# TWAP and VWAP total penalties (negative rewards)
twap_total_penalty = 0.00025039198867976665
vwap_total_penalty = 0.00018039198867976665

# Data for visualization
labels = ['TWAP', 'VWAP', 'RL Model']
rewards = [-twap_total_penalty, -vwap_total_penalty, rl_total_reward]

# Convert negative values to positive for visualization (since they represent penalties)
positive_rewards = [abs(reward) for reward in rewards]

# Highlighting the RL model for better visualization
colors = ['gray', 'gray', 'green']

# Create the bar chart
plt.figure(figsize=(10, 6))
plt.bar(labels, positive_rewards, color=colors, edgecolor='black')
plt.xlabel('Trading Strategy', fontsize=14)
plt.ylabel('Reward (Lower Penalty is Better)', fontsize=14)
plt.title('Comparison of Trading Strategies: TWAP, VWAP, and RL Model', fontsize=16)

# Annotate the bars with the actual penalty values for better clarity
for i, v in enumerate(rewards):
    plt.text(i, abs(v) + 0.00001, f"{v:.6f}", ha='center', va='bottom', fontsize=12, color='black')

plt.grid(axis='y', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()

# Slippage and Market Impact Over Time (Example Data)
time_steps = np.arange(1, 391)  # Assume 390 minutes of trading
twap_slippage = np.random.uniform(0.0001, 0.00025, size=len(time_steps))  # Replace with actual slippage data
vwap_slippage = np.random.uniform(0.0001, 0.00018, size=len(time_steps))
rl_slippage = np.random.uniform(0.00005, 0.00015, size=len(time_steps))

# Plot Slippage Over Time
plt.figure(figsize=(12, 6))
plt.plot(time_steps, twap_slippage, label='TWAP Slippage', color='blue', linestyle='--')
plt.plot(time_steps, vwap_slippage, label='VWAP Slippage', color='orange', linestyle='-.')
plt.plot(time_steps, rl_slippage, label='RL Model Slippage', color='green', linestyle='-')
plt.xlabel('Time Step (Minutes)', fontsize=14)
plt.ylabel('Slippage', fontsize=14)
plt.title('Slippage Over Time for Different Trading Strategies', fontsize=16)
plt.legend()
plt.grid(axis='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()

# Plot Market Impact Over Time (Example Data)
twap_market_impact = np.random.uniform(0.0001, 0.0003, size=len(time_steps))  # Replace with actual market impact data
vwap_market_impact = np.random.uniform(0.0001, 0.0002, size=len(time_steps))
rl_market_impact = np.random.uniform(0.00005, 0.0001, size=len(time_steps))

plt.figure(figsize=(12, 6))
plt.plot(time_steps, twap_market_impact, label='TWAP Market Impact', color='blue', linestyle='--')
plt.plot(time_steps, vwap_market_impact, label='VWAP Market Impact', color='orange', linestyle='-.')
plt.plot(time_steps, rl_market_impact, label='RL Model Market Impact', color='green', linestyle='-')
plt.xlabel('Time Step (Minutes)', fontsize=14)
plt.ylabel('Market Impact', fontsize=14)
plt.title('Market Impact Over Time for Different Trading Strategies', fontsize=16)
plt.legend()
plt.grid(axis='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()

# Cumulative Reward Over Time (Example Data)
rl_cumulative_reward = np.cumsum(np.random.uniform(-0.00005, -0.0001, size=len(time_steps)))  # Replace with actual reward data

plt.figure(figsize=(12, 6))
plt.plot(time_steps, rl_cumulative_reward, label='RL Model Cumulative Reward', color='green', linestyle='-')
plt.xlabel('Time Step (Minutes)', fontsize=14)
plt.ylabel('Cumulative Reward', fontsize=14)
plt.title('Cumulative Reward Over Time for RL Model', fontsize=16)
plt.legend()
plt.grid(axis='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()
