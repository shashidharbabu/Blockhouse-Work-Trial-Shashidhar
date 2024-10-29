import gym
from gym import spaces
import numpy as np
import pandas as pd
from benchmark_costs_script import Benchmark
class TradingEnv(gym.Env):
    """
    Custom Trading Environment for Reinforcement Learning
    """
    def __init__(self, data):
        super(TradingEnv, self).__init__()
        self.remaining_inventory = 1000  # Start with 1000 shares
        self.current_step = 0

        # Load market data
        self.data = data.reset_index(drop=True)
        self.total_timesteps = len(self.data)

        # Action Space: Number of shares to sell (continuous, between 0 and remaining inventory)
        # self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.action_space = spaces.Box(low=0.00, high=1.00, shape=(1,), dtype=np.float32)

        self.benchmark = Benchmark(self.data)
        # Observation Space: State features described above
        self.observation_space = spaces.Box(
            low=-1, high=1, shape=(len(self._get_state(0)),), dtype=np.float32
        )

        # Internal state
        
    def _get_state(self, step):
        """
        Get the current state representation.
        """
        row = self.data.iloc[step]

        state = [
            self.remaining_inventory,  # Remaining inventory
            step / self.total_timesteps,  # Elapsed time as a fraction of the trading day
            row['bid_price_1'],
            row['ask_price_1'],
            row['volume'],
            row['bid_ask_spread'],
            row['ma_short'],
            row['ma_long'],
            row['rolling_volatility'],
            row['hour'],
            row['minute'],
        ]

        # Order book depth: Adding bid and ask sizes
        state += row[['bid_size_1', 'bid_size_2', 'bid_size_3', 'bid_size_4', 'bid_size_5']].tolist()
        state += row[['ask_size_1', 'ask_size_2', 'ask_size_3', 'ask_size_4', 'ask_size_5']].tolist()

        return state

    def reset(self):
        """
        Reset the environment to initial state.
        """
        self.remaining_inventory = 1000  # Reset inventory to 1000 shares
        self.current_step = 0  # Start from the beginning of the data
        return np.array(self._get_state(self.current_step), dtype=np.float32)
    
    def step(self, action):
    # Ensure action value is between 0 and 1
        fraction_to_sell = max(min(action, 1.0), 0.0)
        print(f"Action Value: {action}")

        # Calculate the number of shares to sell
        shares_to_sell = np.round(fraction_to_sell * self.remaining_inventory)

        # Ensure the number of shares is valid
        shares_to_sell = min(max(shares_to_sell, 0), self.remaining_inventory)
        
        # Reduce inventory by the number of shares sold
        self.remaining_inventory -= shares_to_sell
            
        # Get the current timestamp
        current_timestamp = self.data.iloc[self.current_step]['timestamp']
        
        # Store additional information
        info = {'timestamp': current_timestamp}

        # Move to the next step (time progresses by one minute)
        self.current_step += 1

        # Extract the current market data row (if within range)
        if self.current_step < len(self.data):
            row = self.data.iloc[self.current_step]

        # Calculate slippage and market impact
        alpha = 4.439584265535017e-06
        slippage_penalty, market_impact_penalty = self.benchmark.compute_components(alpha, shares_to_sell, self.current_step)

        # Apply penalty if any inventory remains at the end of the day
        remaining_penalty = 0.01 * self.remaining_inventory if self.current_step == self.total_timesteps - 1 else 0

        
        # Calculate total reward (negative value for penalties)
        reward = - (slippage_penalty + market_impact_penalty + remaining_penalty)

        # Check if we have reached the end of the trading day
        done = self.current_step >= self.total_timesteps - 1

        # Get the next state
        state = self._get_state(self.current_step)
        # print(f"State at step {self.current_step}: {state}")
        processed_state = []
        for value in state:
            # If the value is a numpy array (like array([0.], dtype=float32)), extract the first value
            if isinstance(value, np.ndarray):
                processed_state.append(value.item())
            # Convert other types like np.float64 or np.int64 to float
            elif isinstance(value, (np.float64, np.int64)):
                processed_state.append(float(value))
            else:
                processed_state.append(value)
        
        # Convert the processed state to a numpy array with float32 type
        state = np.array(processed_state, dtype=np.float32)

        
        # state = np.array(state, dtype=np.float32)

        return state, reward, done, info

    def render(self, mode='human'):
        """
        Render the environment.
        """
        print(f"Step: {self.current_step}, Remaining Inventory: {self.remaining_inventory}, Action Taken: Sell Shares")

# Create an instance of the environment with the cleaned dataset
# data = pd.read_csv("/Users/shashidharbabu/Documents/07. Projects/Blockhouse /Blockhouse-Work-Trial/data/final_data.csv")
# merged_bid_ask_data = pd.DataFrame(data)

# env = TradingEnv(merged_bid_ask_data)

# # Test resetting and taking a step in the environment
# initial_state = env.reset()
# action = np.array([0.1], dtype=np.float32)  # Example action: sell 10% of remaining inventory
# next_state, reward, done, _ = env.step(action)

# initial_state, next_state, reward, done

# # Print the initial state, next state, reward, and done flag
# print("Initial State:", initial_state)
# print("Next State:", next_state)
# print("Reward:", reward)
# print("Done:", done)
