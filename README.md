# Blockhouse Work Trial: Reinforcement Learning for Optimized Trading Schedule

## Project Overview
This project aims to develop a reinforcement learning (RL) model to solve a trading problem involving the optimized execution of 1000 shares of AAPL over one trading day. The model's objective is to minimize transaction costs by strategically splitting trades over 390 minutes (one trading day). The key features include data analysis, model training using the Soft Actor-Critic (SAC) algorithm, benchmark comparison, and AWS deployment to facilitate real-time inference. 

## Table of Contents
1. [Project Structure](#project-structure)
2. [Getting Started](#getting-started)
3. [Key Components](#key-components)
4. [Model Training and Fine-Tuning](#model-training-and-fine-tuning)
5. [Benchmarking and Back-Testing](#benchmarking-and-back-testing)
6. [AWS Deployment](#aws-deployment)
7. [Running Locally](#running-locally)
8. [Challenges Faced](#challenges-faced)
9. [Results](#results)
10. [References](#references)

## Project Structure
```
BLOCKHOUSE-WORK-TRIAL/
├── .vscode/                              # VS Code configuration files
├── code/                                  # Directory for Python scripts
│   ├── __pycache__/                       # Compiled Python cache files
│   ├── deploy_model_dir/                  # Model deployment related directory
│   ├── model/                             # Directory for storing model files
│   ├── sac_model_directory/               # Directory containing SAC model files
│   ├── sac_trading_model/                 # Directory containing trained SAC trading model files
│   ├── benchmark_costs_script.py          # Script for calculating benchmark trading costs
│   ├── benchmarks.py                      # Script defining benchmark trading strategies
│   ├── cust_trading_env.py                # Custom trading environment definition
│   ├── deploy.py                          # Script to deploy model to AWS SageMaker
│   ├── deployment_testing.py              # Script for testing deployed model
│   ├── generate_trading_schedule.py       # Script to generate trading schedule using trained model
│   ├── invoke_endpoint.py                 # Script to invoke AWS SageMaker endpoint
│   ├── model_test.py                      # Script for local model testing
│   ├── model_train.py                     # Model training script using SAC algorithm
│   ├── model.tar.gz                       # Archived trained model file for deployment
│   ├── sac_trading_model.tar.gz           # Archived SAC model for deployment
│   ├── sac_trading_model.zip              # Zipped version of the SAC model
│   ├── sagemaker_testing.py               # Script to test the SageMaker endpoint
│   ├── server.py                          # Flask server for local model inference
│   ├── viz.py                             # Visualization script for benchmarking results
├── data/                                  # Directory for data files
│   └── EDA/                               # Directory for EDA scripts and notebooks
│       └── data.ipynb                     # Jupyter notebook for exploratory data analysis (EDA)
├── rl_env/                                # Python virtual environment for the project
├── numpy_layer.zip                        # Zipped NumPy layer for AWS Lambda
├── README.md                              # Project documentation file
└── requirements.txt                       # Project dependencies
```

## Getting Started
To get started with this project, clone the repository and set up a Python environment. You can use the virtual environment already provided (`rl_env`) or create your own.

### Prerequisites
- Python 3.8+
- AWS CLI
- Boto3
- Anaconda
- Pandas, Numpy, Matplotlib, Seaborn, Flask, Stable Baselines3
- SAC Algorithm

### Installation
1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd BLOCKHOUSE-WORK-TRIAL
   ```
2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create and activate the virtual environment:
   ```bash
   conda env create -f rl_env
   conda activate rl_env
   ```

## Key Components
### 1. Data Analysis and Feature Engineering
- **EDA and Feature Engineering**: Conducted exploratory data analysis to understand the data structure and identified relationships between key features, such as bid/ask price, spread, volume, and volatility.
- **Feature Engineering**: Created derived features like bid-ask spread, moving averages, and rolling volatility to enhance model learning.

### 2. Model Training and Fine-Tuning
- The RL model was trained using the Soft Actor-Critic (SAC) algorithm, a state-of-the-art off-policy method to deal with the continuous action space.
- **Model Architecture**: The SAC algorithm uses an actor-critic approach to optimize a policy by balancing exploration and exploitation, utilizing stochastic gradient descent.
- **Fine-Tuning**: Hyperparameters like learning rate, batch size, entropy coefficient, etc., were tuned for optimal performance.

### 3. Benchmarking and Back-Testing
- **TWAP and VWAP Strategies**: Benchmarked RL model performance against traditional strategies such as TWAP (Time-Weighted Average Price) and VWAP (Volume-Weighted Average Price).
- **Back-Testing**: Tested the trained model on historical market data to evaluate its effectiveness in reducing transaction costs.
- **Results Visualization**: Created visual comparisons of penalties (slippage and market impact) between the RL model, TWAP, and VWAP.

### 4. AWS Deployment
- The trained model was deployed using **AWS SageMaker** as a real-time endpoint for live trade data prediction.
- **Lambda Integration**: AWS Lambda function used to interact with the model and serve the inference results as a JSON API response.
- **Instance Type Selection**: To avoid timeouts, higher computational power was selected for AWS instances.

### 5. Running Locally
- A Flask server (`server.py`) was built to run the model locally, accepting inputs (ticker, shares, and time horizon) and outputting an optimized trading schedule.

## Model Training and Fine-Tuning
The model training was done using the SAC algorithm with hyperparameter tuning. The following aspects were considered:

- **Policy Network**: Used a stochastic policy to allow for exploration and prevent the model from getting stuck in local optima.
- **Training Environment**: A custom environment (`cust_trading_env.py`) was developed using the OpenAI Gym interface to simulate trading over 390 minutes.
- **Hyperparameter Tuning**: Tuned parameters like learning rate, reward scaling, etc., to optimize the model's ability to reduce market impact.

## Benchmarking and Back-Testing
The model was tested against TWAP and VWAP benchmarks to evaluate its performance:
- **TWAP**: Spread trades equally over time, regardless of market conditions.
- **VWAP**: Executes trades based on the volume of each interval.
- **Results**: The RL model outperformed both strategies, as seen in back-testing results visualized with line charts and cumulative reward comparisons.

## AWS Deployment
- **Model Deployment**: Deployed on AWS SageMaker using the `deploy.py` script and configured as a real-time inference endpoint.
- **Endpoint Testing**: Invoked the endpoint using `sagemaker_testing.py` to ensure live data is being handled as expected.
- **Lambda Function**: An AWS Lambda function was used to create an API, providing users the ability to interact with the deployed model and get responses in JSON format.

## Running Locally
### Commands to Run Key Components
- **Model Training**:
  ```bash
  python code/model_train.py
  ```
- **Model Testing**:
  ```bash
  python code/model_test.py
  ```
- **Generate Trading Schedule**:
  ```bash
  python code/generate_trading_schedule.py
  ```
- **Run Flask Server**:
  ```bash
  python code/server.py
  ```

## Challenges Faced
1. **Training Complexity**: The large action space required significant computation power, and balancing exploration and exploitation posed challenges.
2. **AWS Timeouts**: Multiple endpoint timeouts were encountered during initial deployment. This was mitigated by increasing the instance size and optimizing request payloads.
3. **Dependency Management**: Required packaging dependencies, such as NumPy, for Lambda execution.

## Results
The RL model successfully optimized the trading schedule, leading to a lower penalty (better cumulative reward) compared to TWAP and VWAP strategies. The back-testing phase confirmed that the model was able to learn an effective policy for minimizing market impact and slippage.

## References
- **Stable Baselines3 Documentation**: [https://stable-baselines3.readthedocs.io](https://stable-baselines3.readthedocs.io)
- **AWS SageMaker Documentation**: [https://docs.aws.amazon.com/sagemaker](https://docs.aws.amazon.com/sagemaker)
- **Soft Actor-Critic Paper**: [https://arxiv.org/abs/1812.05905](https://arxiv.org/abs/1812.05905)
- **Boto3 Library Documentation**: [https://boto3.amazonaws.com/v1/documentation](https://boto3.amazonaws.com/v1/documentation)

---
For further information or questions, refer to the `README.md` or contact the author.
