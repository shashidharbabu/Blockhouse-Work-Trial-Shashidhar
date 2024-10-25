from cust_trading_env import TradingEnv
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
import pandas as pd

data = pd.read_csv("/Users/shashidharbabu/Documents/07. Projects/Blockhouse /Blockhouse-Work-Trial/data/final_data.csv")
merged_bid_ask_data = pd.DataFrame(data)

split_ratio = 0.8  # 80% training, 20% testing
split_index = int(len(data) * split_ratio)

# Split the dataset into training and testing sets
train_data = data.iloc[:split_index]
merged_bid_ask_data_train = pd.DataFrame(train_data)



# Wrap the TradingEnv in a vectorized environment for training
train_env = DummyVecEnv([lambda: TradingEnv(merged_bid_ask_data)])

# Define the SAC model with MlpPolicy
model = SAC(
    "MlpPolicy",        # Policy type: Multilayer Perceptron (MLP)
    train_env,            # Environment for training
    verbose=1,          # Verbosity level (to display training info)
    learning_rate=0.001,  # Learning rate for the model
    gamma=0.99,           # Discount factor
    batch_size=64         # Batch size for training
)

# Train the model
timesteps = 50000  # Set number of timesteps to train
model.learn(total_timesteps=timesteps)
print("Training completed successfully.")


# Save the trained model for later use
model.save("/Users/shashidharbabu/Documents/07. Projects/Blockhouse /Blockhouse-Work-Trial/code/sac_trading_model")
