import pandas as pd
from benchmark_costs_script import Benchmark

# Load the dataset
data = pd.read_csv("/Users/shashidharbabu/Documents/07. Projects/Blockhouse /Blockhouse-Work-Trial/data/final_data.csv")
merged_bid_ask_data = pd.DataFrame(data)

# Create an instance of the Benchmark class
benchmark = Benchmark(merged_bid_ask_data)

# Set parameters for TWAP and VWAP
initial_inventory = 1000  # The total number of shares to be sold
preferred_timeframe = 390  # Number of time steps (e.g., representing a full trading day)

# Generate TWAP trades
twap_trades = benchmark.get_twap_trades(merged_bid_ask_data, initial_inventory, preferred_timeframe)

# Generate VWAP trades
vwap_trades = benchmark.get_vwap_trades(merged_bid_ask_data, initial_inventory, preferred_timeframe)

# Output the trades for verification
print("TWAP Trades:")
print(twap_trades.head())

print("\nVWAP Trades:")
print(vwap_trades.head())

####################################
# Simulate TWAP strategy and calculate slippage and market impact
twap_slippage, twap_market_impact = benchmark.simulate_strategy(twap_trades, data, preferred_timeframe)
print("TWAP Slippage:", twap_slippage)
print("TWAP Market Impact:", twap_market_impact)

# Simulate VWAP strategy and calculate slippage and market impact
vwap_slippage, vwap_market_impact = benchmark.simulate_strategy(vwap_trades, data, preferred_timeframe)
print("VWAP Slippage:", vwap_slippage)
print("VWAP Market Impact:", vwap_market_impact)


rl_total_reward = -0.00014039198867976665 # Replace with your actual RL reward after running the test

# TWAP and VWAP total penalties
twap_total_penalty = sum(twap_slippage) + sum(twap_market_impact)
vwap_total_penalty = sum(vwap_slippage) + sum(vwap_market_impact)

# Print the results for comparison
print("Total Reward using TWAP Strategy:", -twap_total_penalty)
print("Total Reward using VWAP Strategy:", -vwap_total_penalty)
print("Total Reward during Backtesting (RL Model):", rl_total_reward)