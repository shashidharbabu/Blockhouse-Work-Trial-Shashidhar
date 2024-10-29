# test_model.py

from cust_trading_env import TradingEnv
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
import pandas as pd

# Load the dataset
data = pd.read_csv("/Users/shashidharbabu/Documents/07. Projects/Blockhouse /Blockhouse-Work-Trial/data/final_data.csv")
merged_bid_ask_data = pd.DataFrame(data)

# Load the trained model
model = SAC.load("/Users/shashidharbabu/Documents/07. Projects/Blockhouse /Blockhouse-Work-Trial/code/sac_trading_model.zip")

split_ratio = 0.8  # 80% training, 20% testing
split_index = int(len(data) * split_ratio)

# Split the dataset
train_data = data.iloc[:split_index]
test_data = data.iloc[split_index:]



# Create a test environment
test_env = DummyVecEnv([lambda: TradingEnv(test_data)])

# Reset the environment and initialize variables
obs = test_env.reset()
done = False
total_reward = 0
time_steps = 0  # Keep track of the number of time steps

# Run the testing loop until the end of the day (or until done)
while not done:
    # Get action from the trained model
    action, _states = model.predict(obs, deterministic=True)

    # Normalize action from [-1, 1] to [0, 1] if necessary
    action_normalized = (action[0] + 1) / 2  # Now action_normalized is between 0 and 1

    # Take a step in the environment
    obs, reward, done, info = test_env.step([action_normalized])
    total_reward += reward[0]  # Since reward is returned as a list, take the first element

    time_steps += 1

    # Stop if we reach the 390-minute horizon
    if time_steps >= 390:
        done = True

# Output the total reward for evaluation
print(f"Total Reward during Backtesting: {total_reward}")
