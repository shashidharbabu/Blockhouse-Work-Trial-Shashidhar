from sagemaker.pytorch import PyTorchModel
import sagemaker

# S3 URI for the model artifact
model_data = 's3://blockhouse-work-trial-models/model.tar.gz'

# Role with SageMaker permissions
role = 'arn:aws:iam::084828570399:role/SageMakerExecutionRole'

# Set up the SageMaker session
sagemaker_session = sagemaker.Session()

# Define the PyTorch model for deployment
model = PyTorchModel(
    model_data=model_data,
    role=role,
    entry_point='generate_trading_schedule.py',  # This is the inference script
    framework_version='1.8.0',  # Set the PyTorch version to match your setup
    py_version='py3',
    sagemaker_session=sagemaker_session
)

# Deploy the model to create a SageMaker endpoint
predictor = model.deploy(
    initial_instance_count=1,
    instance_type='ml.m5.large',  # Choose the instance type that suits your requirements
    endpoint_name='blockhouse-trial-endpoint'
)
