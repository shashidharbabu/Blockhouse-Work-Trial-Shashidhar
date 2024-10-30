import json
import os
import pandas as pd
import numpy as np
from cust_trading_env import TradingEnv
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from flask import Flask, request, jsonify

# Initialize Flask app
app = Flask(__name__)

# Load the trained model from the S3 path
MODEL_PATH = os.getenv('MODEL_PATH', '/Users/shashidharbabu/Documents/07. Projects/Blockhouse /Blockhouse-Work-Trial/code/sac_trading_model.zip')  # Path to your model
model = SAC.load(MODEL_PATH)

# Load the dataset (for state extraction)
DATA_PATH = os.getenv('DATA_PATH', '/Users/shashidharbabu/Documents/07. Projects/Blockhouse /Blockhouse-Work-Trial/data/final_data.csv')  # Path to your dataset
data = pd.read_csv(DATA_PATH)

# Define API endpoint to generate trading schedule
@app.route('/generate_schedule', methods=['POST'])
def generate_schedule():
    # Parse incoming JSON request
    input_data = request.get_json()
    ticker = input_data.get("ticker", "AAPL")
    total_shares = int(input_data.get("total_shares", 1000))
    time_horizon = int(input_data.get("time_horizon", 390))

    # Create a new trading environment instance
    test_env = DummyVecEnv([lambda: TradingEnv(data)])

    # Initialize variables for running the model in the environment
    obs = test_env.reset()
    done = False
    trade_schedule = []
    remaining_inventory = total_shares
    total_reward = 0
    time_steps = 0

    # Run the trading loop to generate a schedule
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

        # Stop if we reach the time horizon or have no inventory left
        if time_steps >= time_horizon or remaining_inventory <= 0:
            done = True

    # Convert trade_schedule to a DataFrame for better presentation
    trade_schedule_df = pd.DataFrame(trade_schedule)
    response = trade_schedule_df.to_dict(orient='records')

    return jsonify(response)

# Run the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
