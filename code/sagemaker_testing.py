import boto3
import json
import pandas as pd
import numpy as np
from botocore.config import Config

# Load the dataset for preprocessing
DATA_PATH = '/Users/shashidharbabu/Documents/07. Projects/Blockhouse /Blockhouse-Work-Trial/data/final_data.csv'
data = pd.read_csv(DATA_PATH)

config = Config(
    connect_timeout=600,  # Increase the timeout if necessary
    read_timeout=600      # Increase the timeout if necessary
)

# Initialize the SageMaker runtime client
client = boto3.client('sagemaker-runtime', region_name='us-east-2', config=config)

# Define your SageMaker endpoint name here
ENDPOINT_NAME = 'endpoint-for-deployment-v2'

# Preprocess the input to create the state for the model
def create_initial_state(step, total_shares):
    row = data.iloc[step]
    remaining_inventory = total_shares
    total_timesteps = len(data)

    state = [
        remaining_inventory,
        step / total_timesteps,
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

    # Adding order book depth: bid and ask sizes
    state += row[['bid_size_1', 'bid_size_2', 'bid_size_3', 'bid_size_4', 'bid_size_5']].tolist()
    state += row[['ask_size_1', 'ask_size_2', 'ask_size_3', 'ask_size_4', 'ask_size_5']].tolist()
    # Convert the state to a float32 list
    return list(np.array(state, dtype=np.float32))

# Define the input data
initial_step = 0
total_shares = 1000
batch_size = 50 # Define a smaller batch size

responses = []

# Loop through the data and send batch requests
for start_step in range(0, len(data), batch_size):
    # Create a batch of inputs
    batch_states = []
    for step in range(start_step, min(start_step + batch_size, len(data))):
        # Convert the state to a list of standard Python floats
        state = list(map(float, create_initial_state(step, total_shares)))
        batch_states.append(state)

    # Prepare the payload to send to SageMaker
    input_data = {
        "inputs": {
            "state": batch_states,  # Send a list of states as a batch
            "total_shares": total_shares,
            "time_horizon": 390,
        }
    }

    # Convert input data to JSON
    payload = json.dumps(input_data)

    # Invoke the endpoint
    try:
        response = client.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType="application/json",
            Body=payload
        )

        # Parse the response and store it
        result = json.loads(response['Body'].read().decode())
        responses.append(result)

    except Exception as e:
        print(f"An error occurred while processing the batch from step {start_step} to {start_step + batch_size}: {str(e)}")

# Process the combined responses
print("Responses from SageMaker Endpoint:")
for response in responses:
    print(response)
