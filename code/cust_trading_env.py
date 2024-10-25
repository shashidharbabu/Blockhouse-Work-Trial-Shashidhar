import gym
from gym import spaces
import numpy as np
import pandas as pd

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
        self.action_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)

        # Observation Space: State features described above
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(self._get_state(0)),), dtype=np.float32
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
        """
        Take an action and return the next state, reward, done flag, and info.
        """
        # Calculate number of shares to sell, constrained by remaining inventory
        shares_to_sell = (action[0] + 1) * self.remaining_inventory / 2  # Scale to ensure impactful trades
        shares_to_sell = min(max(shares_to_sell, 0), self.remaining_inventory)

        # Update remaining inventory
        self.remaining_inventory -= shares_to_sell

        # Move to the next step (time progresses by one minute)
        self.current_step += 1

        # Extract the current market data row
        row = self.data.iloc[self.current_step]

        # Calculate reward components with adjusted scaling
        slippage_penalty = row['bid_ask_spread'] / 2  # Reduce the effect of slippage penalty by dividing by 2
        market_impact_penalty = 30 * (shares_to_sell / (row['volume'] + 1e-5))
        
        # Apply penalty if any inventory remains at the end of the day
        remaining_penalty = 20 * self.remaining_inventory if self.current_step == self.total_timesteps - 1 else 0

        # Calculate total reward (negative value for penalties)
        reward = - (slippage_penalty + market_impact_penalty + remaining_penalty)

        # Check if we have reached the end of the trading day
        done = self.current_step >= self.total_timesteps - 1

        # Get the next state
        state = self._get_state(self.current_step)

        # Print the reward components for debugging
        print(f"Slippage Penalty: {-slippage_penalty}, Market Impact Penalty: {-market_impact_penalty}, Remaining Penalty: {-remaining_penalty}, Current Step: {self.current_step}")

        return np.array(state, dtype=np.float32), reward, done, {}

    def render(self, mode='human'):
        """
        Render the environment.
        """
        print(f"Step: {self.current_step}, Remaining Inventory: {self.remaining_inventory}, Action Taken: Sell Shares")

# Create an instance of the environment with the cleaned dataset
data = pd.read_csv("/Users/shashidharbabu/Documents/07. Projects/Blockhouse /Blockhouse-Work-Trial/data/final_data.csv")
merged_bid_ask_data = pd.DataFrame(data)

env = TradingEnv(merged_bid_ask_data)

# Test resetting and taking a step in the environment
initial_state = env.reset()
action = np.array([0.1], dtype=np.float32)  # Example action: sell 10% of remaining inventory
next_state, reward, done, _ = env.step(action)

initial_state, next_state, reward, done

# Print the initial state, next state, reward, and done flag
print("Initial State:", initial_state)
print("Next State:", next_state)
print("Reward:", reward)
print("Done:", done)
