import sagemaker
import json

# Set up the SageMaker session
sagemaker_session = sagemaker.Session()

# Connect to the deployed endpoint
predictor = sagemaker.Predictor(endpoint_name='blockhouse-trial-endpoint', sagemaker_session=sagemaker_session)

# Input data for testing
input_data = {
    "ticker": "AAPL",
    "number_of_shares": 1000,
    "time_horizon": 390  # The trading day in minutes
}

# Convert to JSON string
payload = json.dumps(input_data)

# Make a prediction
result = predictor.predict(payload)
print(result)
