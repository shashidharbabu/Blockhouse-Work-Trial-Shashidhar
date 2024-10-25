import numpy as np
import pandas as pd

data = pd.read_csv("/Users/shashidharbabu/Documents/07. Projects/Blockhouse /Blockhouse-Work-Trial/data/final_data.csv")
merged_bid_ask_data = pd.DataFrame(data)


# Calculate TWAP and VWAP schedules for comparison
# TWAP (Time-Weighted Average Price) Calculation
num_intervals = len(merged_bid_ask_data)  # Assuming one entry per minute, e.g., 390 minutes in a trading day
twap_shares_per_interval = 1000 / num_intervals  # Assuming selling 1000 shares across all time intervals equally

# VWAP (Volume-Weighted Average Price) Calculation
total_volume = merged_bid_ask_data['volume'].sum()
vwap_allocation = merged_bid_ask_data['volume'] / total_volume * 1000  # Allocating shares proportionally to volume

# Initialize variables for tracking TWAP and VWAP rewards
twap_total_reward = 0
vwap_total_reward = 0

# Backtest TWAP strategy
remaining_inventory = 1000
for idx in range(num_intervals):
    shares_to_sell = twap_shares_per_interval
    row = merged_bid_ask_data.iloc[idx]
    
    # Calculate penalties similar to reward function in RL model
    slippage_penalty = row['bid_ask_spread']/2
    market_impact_penalty = 30 *(shares_to_sell / (row['volume'] + 1e-5))
    remaining_penalty = remaining_inventory - shares_to_sell if idx == num_intervals - 1 else 0
    

    
    # Update TWAP reward
    twap_reward = - (slippage_penalty + market_impact_penalty + remaining_penalty)
    twap_total_reward += twap_reward
    remaining_inventory -= shares_to_sell

# Backtest VWAP strategy
remaining_inventory = 1000
for idx in range(num_intervals):
    shares_to_sell = vwap_allocation.iloc[idx]
    row = merged_bid_ask_data.iloc[idx]
    
    # Calculate penalties similar to reward function in RL model
    slippage_penalty = row['bid_ask_spread']
    market_impact_penalty = shares_to_sell / (row['volume'] + 1e-5)
    remaining_penalty = remaining_inventory - shares_to_sell if idx == num_intervals - 1 else 0
    
    # Update VWAP reward
    vwap_reward = - (slippage_penalty + market_impact_penalty + remaining_penalty)
    vwap_total_reward += vwap_reward
    remaining_inventory -= shares_to_sell

# Print the comparison results
# print("Total Reward during Backtesting (RL Model):", total_rewards)
print("Total Reward using TWAP Strategy:", twap_total_reward)
print("Total Reward using VWAP Strategy:", vwap_total_reward)