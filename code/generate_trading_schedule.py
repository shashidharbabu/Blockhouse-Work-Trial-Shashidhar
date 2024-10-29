from cust_trading_env import TradingEnv
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
import pandas as pd

# Load data
data = pd.read_csv("/Users/shashidharbabu/Documents/07. Projects/Blockhouse /Blockhouse-Work-Trial/data/final_data.csv")
merged_bid_ask_data = pd.DataFrame(data)

# Load the trained model
model = SAC.load("/Users/shashidharbabu/Documents/07. Projects/Blockhouse /Blockhouse-Work-Trial/code/sac_trading_model.zip")

# Create test environment
test_env = DummyVecEnv([lambda: TradingEnv(merged_bid_ask_data)])

# Reset the environment and initialize variables
obs = test_env.reset()
done = False
trade_schedule = []
total_shares = 1000  # Total shares to sell
remaining_inventory = total_shares
total_reward = 0
time_steps = 0  # Keep track of the time steps

# Run the trading loop until all shares are sold or trading day ends
while not done and remaining_inventory > 0:
    # Get the action from the model
    action, _states = model.predict(obs)

    # Normalize action to be between 0 and 1
    action_normalized = (action[0] + 1) / 2  # Converts [-1, 1] to [0, 1]

    # Cap the shares to sell to ensure gradual selling
    max_sell_fraction = 0.1  # Limit to selling 10% of remaining inventory per minute
    fraction_to_sell = min(action_normalized, max_sell_fraction)

    # Denormalize action to represent shares to be sold
    shares_to_sell = int(fraction_to_sell * remaining_inventory)

    # Ensure shares_to_sell is within a valid range
    shares_to_sell = min(max(shares_to_sell, 1), remaining_inventory)

    # Take a step in the environment
    obs, reward, done, info = test_env.step([shares_to_sell])
    total_reward += reward[0]  # reward is returned as a list, take the first element
    current_timestamp = info[0].get("timestamp")

    # Update the remaining inventory
    remaining_inventory -= shares_to_sell
    time_steps += 1

    # Add the trade to the schedule
    trade_schedule.append({
        'timestamp': current_timestamp,
        'share_size': shares_to_sell
    })

    # Stop if we reach the 390-minute horizon or have no inventory left
    if time_steps >= 390 or remaining_inventory <= 0:
        done = True

# Output the total reward
print(f"Total Reward: {total_reward}")

# Convert trade_schedule to a DataFrame for better presentation or save it as CSV
trade_schedule_df = pd.DataFrame(trade_schedule)
print(trade_schedule_df)
