from stable_baselines3 import SAC, PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import pandas as pd
from cust_trading_env import TradingEnv
# Load the dataset
import numpy as np
from stable_baselines3.common.noise import NormalActionNoise


data = pd.read_csv("/Users/shashidharbabu/Documents/07. Projects/Blockhouse /Blockhouse-Work-Trial/data/final_data.csv")
merged_bid_ask_data = pd.DataFrame(data)

# Split the dataset into training and testing sets
split_ratio = 0.8  # 80% training, 20% testing
split_index = int(len(data) * split_ratio)

# Split the dataset
train_data = data.iloc[:split_index]
test_data = data.iloc[split_index:]

# Create training environment
vec_env = DummyVecEnv([lambda: TradingEnv(train_data)])


n_actions = vec_env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.5 * np.ones(n_actions))


#Defining PPO model


# model = PPO(
#     "MlpPolicy",
#     vec_env,
#     verbose=1,
#     learning_rate=0.001,
#     gamma=0.99,
#     batch_size=64
# )

# # Define the SAC model
model = SAC(
    "MlpPolicy",        # Policy type: Multilayer Perceptron (MLP)
    vec_env,            # Vectorized environment for training
    verbose=1,          # Verbosity level (1 for showing training info)
    learning_rate=0.0005,  # Learning rate for the SAC algorithm
    gamma=0.99,           # Discount factor
    batch_size = 128,         # Batch size for training
    ent_coef = 0.5,
    action_noise=action_noise  # Add noise for better exploration

)

# Train the model
timesteps = 10000  # Set the number of timesteps for training
model.learn(total_timesteps=timesteps)

# Save the trained model
model.save("/Users/shashidharbabu/Documents/07. Projects/Blockhouse /Blockhouse-Work-Trial/code/sac_trading_model")

print("Training completed successfully.")

# # Test the trained model
# # Load the trained model
# model = SAC.load("/Users/shashidharbabu/Documents/07. Projects/Blockhouse /Blockhouse-Work-Trial/code/sac_trading_model")

# # Reset the environment to the beginning for evaluation
# obs = vec_env.reset()

# # Run the model in the environment to simulate and evaluate
# total_rewards = 0
# done = False

# while not done:
#     # Get action from the trained model
#     action, _states = model.predict(obs, deterministic=True)  # Use deterministic actions for evaluation
#     print(f"Action taken by model: {action}")  # Debugging to check action variability
#     # Take action in the environment
#     obs, reward, done, _ = vec_env.step(action)
#     total_rewards += reward

# # Print the accumulated rewards for evaluation
# print("Total Reward during Backtesting:", total_rewards)












# from cust_trading_env import TradingEnv
# from stable_baselines3 import SAC
# from stable_baselines3.common.vec_env import DummyVecEnv
# import pandas as pd

# data = pd.read_csv("/Users/shashidharbabu/Documents/07. Projects/Blockhouse /Blockhouse-Work-Trial/data/final_data.csv")
# merged_bid_ask_data = pd.DataFrame(data)

# split_ratio = 0.8  # 80% training, 20% testing
# split_index = int(len(data) * split_ratio)

# # Split the dataset into training and testing sets
# train_data = data.iloc[:split_index]
# merged_bid_ask_data_train = pd.DataFrame(train_data)



# # Wrap the TradingEnv in a vectorized environment for training
# train_env = DummyVecEnv([lambda: TradingEnv(merged_bid_ask_data_train)])

# # Define the SAC model with MlpPolicy
# model = SAC(
#     "MlpPolicy",        # Policy type: Multilayer Perceptron (MLP)
#     train_env,            # Environment for training
#     verbose=1,          # Verbosity level (to display training info)
#     learning_rate=0.001,  # Learning rate for the model
#     gamma=0.99,           # Discount factor
#     batch_size=64         # Batch size for training
# )

# # Train the model
# timesteps = 10000  # Set number of timesteps to train
# model.learn(total_timesteps=timesteps)
# print("Training completed successfully.")


# # Save the trained model for later use
# model.save("/Users/shashidharbabu/Documents/07. Projects/Blockhouse /Blockhouse-Work-Trial/code/sac_trading_model")
