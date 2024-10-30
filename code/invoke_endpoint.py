import boto3
import json
import sys

# Function to invoke the SageMaker endpoint
def invoke_sagemaker_endpoint(endpoint_name, ticker, shares, time_horizon):
    # Set up the SageMaker runtime client
    runtime_client = boto3.client('sagemaker-runtime', region_name='us-east-2')  # Replace 'your-region'

    # Define input parameters
    input_data = {
        "ticker": ticker,
        "shares": shares,
        "time_horizon": time_horizon
    }

    # Invoke the endpoint
    try:
        response = runtime_client.invoke_endpoint(
            EndpointName=endpoint_name,  # Update to your endpoint name
            ContentType="application/json",
            Body=json.dumps(input_data)
        )

        # Parse the response
        result = json.loads(response['Body'].read().decode())
        return result
    except Exception as e:
        print(f"Error invoking the endpoint: {str(e)}")
        return None

# Main function to take command line inputs
if __name__ == "__main__":
    # Ensure all arguments are provided
    if len(sys.argv) != 5:
        print("Usage: python invoke_endpoint.py <endpoint_name> <ticker> <shares> <time_horizon>")
        sys.exit(1)

    # Read inputs
    endpoint_name = sys.argv[1]
    ticker = sys.argv[2]
    shares = int(sys.argv[3])
    time_horizon = int(sys.argv[4])

    # Invoke the endpoint
    output = invoke_sagemaker_endpoint(endpoint_name, ticker, shares, time_horizon)

    # Print the generated trading schedule
    if output:
        print("Generated Trading Schedule:")
        print(json.dumps(output, indent=4))
    else:
        print("Failed to generate trading schedule.")
