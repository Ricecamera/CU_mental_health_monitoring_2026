"""
Deploy MLflow model to SageMaker endpoint.

This script:
1. Downloads a model from MLflow Model Registry
2. Packages it for SageMaker
3. Uploads to S3
4. Creates SageMaker model and endpoint
"""
import io
import os
import sys
import json
import argparse
import tarfile
import tempfile
import pandas as pd

import boto3
import botocore.exceptions
import mlflow

def download_mlflow_model(model_name, model_version, model_stage, mlflow_uri, output_dir):
    """Download model from MLflow Registry"""
    print(f"Connecting to MLflow at {mlflow_uri}")
    mlflow.set_tracking_uri(mlflow_uri)
    
    if model_stage:
        model_uri = f"models:/{model_name}/{model_stage}"
        print(f"Downloading model: {model_name} stage={model_stage}")
    else:
        model_uri = f"models:/{model_name}/{model_version}"
        print(f"Downloading model: {model_name} version={model_version}")
    
    # Download model
    local_path = mlflow.artifacts.download_artifacts(model_uri, dst_path=output_dir)
    print(f"Model downloaded to: {local_path}")
    
    return local_path


def create_model_archive(model_path, inference_script, output_path):
    """Create model.tar.gz for SageMaker"""
    print(f"Creating model archive at {output_path}")
    
    with tarfile.open(output_path, "w:gz") as tar:
        # Add model files (exclude MLflow requirements.txt - use ours instead)
        for item in os.listdir(model_path):
            # Skip MLflow requirements.txt - it has incompatible versions
            if item == 'requirements.txt':
                print(f"Skipping MLflow requirements.txt (using sagemaker_requirements.txt instead)")
                continue
            
            item_path = os.path.join(model_path, item)
            tar.add(item_path, arcname=item)
        
        # Add inference script
        tar.add(inference_script, arcname="code/inference.py")

        setup_py = (
            "from setuptools import setup\n"
            "setup(name='inference', version='1.0.0', py_modules=['inference'])\n"
        )
        setup_py_bytes = setup_py.encode("utf-8")
        info = tarfile.TarInfo(name="code/setup.py")
        info.size = len(setup_py_bytes)
        tar.addfile(info, io.BytesIO(setup_py_bytes))
        print("Added code/setup.py with explicit py_modules=['inference']")

        # Add SageMaker-compatible requirements
        requirements_path = os.path.join(os.path.dirname(inference_script), "sagemaker_requirements.txt")
        if os.path.exists(requirements_path):
            tar.add(requirements_path, arcname="code/requirements.txt")
            print(f"Added sagemaker_requirements.txt as code/requirements.txt")
    
    print(f"Model archive created: {output_path}")


def upload_to_s3(file_path, bucket, key):
    """Upload model archive to S3"""
    print(f"Uploading to s3://{bucket}/{key}")
    s3_client = boto3.client('s3')
    s3_client.upload_file(file_path, bucket, key)
    print("Upload complete")
    return f"s3://{bucket}/{key}"


def create_sagemaker_model(model_name, model_data_url, role, region):
    """Create SageMaker model"""
    print(f"Creating SageMaker model: {model_name}")
    
    sagemaker_client = boto3.client('sagemaker', region_name=region)
    
    account = "121021644041"
    image_uri = f"{account}.dkr.ecr.{region}.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3"
    print(f"Using container image: {image_uri}")
    
    # Delete existing model so we can recreate it pointing to the new artifact URL.
    try:
        sagemaker_client.delete_model(ModelName=model_name)
        print(f"Deleted existing model: {model_name}")
    except Exception:
        pass  # Model did not exist yet
  
    response = sagemaker_client.create_model(
        ModelName=model_name,
        PrimaryContainer={
            'Image': image_uri,
            'ModelDataUrl': model_data_url,
            'Environment': {
                'SAGEMAKER_PROGRAM': 'inference.py',
                'SAGEMAKER_SUBMIT_DIRECTORY': '/opt/ml/model/code',
            }
        },
        ExecutionRoleArn=role
    )
    print(f"Model created: {response['ModelArn']}")
    return response['ModelArn']


def create_endpoint_config(config_name, model_name, instance_type, region, serverless=False, memory_size_mb=None, max_concurrency=None):
    """Create SageMaker endpoint configuration (real-time or serverless)"""
    print(f"Creating endpoint configuration: {config_name}")
    
    sagemaker_client = boto3.client('sagemaker', region_name=region)
    
    try:
        variant_config = {
            'VariantName': 'AllTraffic',
            'ModelName': model_name,
            'InitialVariantWeight': 1.0
        }
        
        if serverless:
            # Serverless Inference configuration
            print(f"Using Serverless Inference: {memory_size_mb}MB, max concurrency {max_concurrency}")
            variant_config['ServerlessConfig'] = {
                'MemorySizeInMB': memory_size_mb,
                'MaxConcurrency': max_concurrency
            }
        else:
            # Real-time endpoint configuration
            print(f"Using real-time endpoint: {instance_type}")
            variant_config['InitialInstanceCount'] = 1
            variant_config['InstanceType'] = instance_type
        
        response = sagemaker_client.create_endpoint_config(
            EndpointConfigName=config_name,
            ProductionVariants=[variant_config]
        )
        print(f"Endpoint config created: {response['EndpointConfigArn']}")
        return response['EndpointConfigArn']
    
    except sagemaker_client.exceptions.ResourceInUse:
        print(f"Endpoint config {config_name} already exists")
        return None


def create_or_update_endpoint(endpoint_name, config_name, region):
    """Create or update SageMaker endpoint"""
    sagemaker_client = boto3.client('sagemaker', region_name=region)

    # Check whether the endpoint already exists and what state it is in.
    endpoint_exists = False
    try:
        desc = sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
        endpoint_exists = True
        status = desc['EndpointStatus']
    except botocore.exceptions.ClientError:
        status = None

    if endpoint_exists and status == 'Failed':
        # SageMaker does not allow update_endpoint on a FAILED endpoint.
        # Delete it and wait for full deletion before recreating.
        print(f"Endpoint is in FAILED state — deleting before recreating: {endpoint_name}")
        try:
            sagemaker_client.delete_endpoint(EndpointName=endpoint_name)
        except botocore.exceptions.ClientError as exc:
            code = exc.response['Error']['Code']
            if code == 'AccessDeniedException':
                print("\n" + "="*70)
                print("PERMISSION ERROR: sagemaker:DeleteEndpoint is missing.")
                print("Add it to your IAM policy, then manually delete the endpoint:")
                print(f"  aws sagemaker delete-endpoint --endpoint-name {endpoint_name} --region {region}")
                print("See IAM_PERMISSIONS.md for the full recommended policy.")
                print("="*70 + "\n")
            raise
        waiter = sagemaker_client.get_waiter('endpoint_deleted')
        waiter.wait(EndpointName=endpoint_name)
        endpoint_exists = False

    if endpoint_exists:
        print(f"Updating existing endpoint: {endpoint_name}")
        response = sagemaker_client.update_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=config_name
        )
        print(f"Endpoint update initiated: {response['EndpointArn']}")
    else:
        print(f"Creating new endpoint: {endpoint_name}")
        response = sagemaker_client.create_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=config_name
        )
        print(f"Endpoint creation initiated: {response['EndpointArn']}")

    print(f"Waiting for endpoint to be in service...")
    waiter = sagemaker_client.get_waiter('endpoint_in_service')
    waiter.wait(EndpointName=endpoint_name)
    print(f"Endpoint {endpoint_name} is now in service!")


def main():
    parser = argparse.ArgumentParser(description='Deploy MLflow model to SageMaker')
    
    # MLflow parameters
    parser.add_argument('--mlflow-uri', required=True, help='MLflow tracking URI')
    parser.add_argument('--model-name', required=True, help='MLflow model name')
    parser.add_argument('--model-version', default='1', help='Model version number')
    parser.add_argument('--model-stage', default=None, help='Model stage (Production/Staging)')
    
    # AWS parameters
    parser.add_argument('--region', default='ap-southeast-1', help='AWS region')
    parser.add_argument('--role', required=True, help='SageMaker execution role ARN')
    parser.add_argument('--s3-bucket', required=True, help='S3 bucket for model artifacts')
    parser.add_argument('--s3-prefix', default='sagemaker/mental-health-models', help='S3 prefix')
    
    # SageMaker parameters
    parser.add_argument('--endpoint-name', default='mental-health-inference', help='Endpoint name')
    parser.add_argument('--instance-type', default='ml.t2.medium', help='Instance type (for real-time)')
    
    # Serverless parameters
    parser.add_argument('--serverless', action='store_true', help='Deploy as serverless endpoint')
    parser.add_argument('--memory-size-mb', type=int, default=2048, 
                        choices=[1024, 2048, 3072, 4096, 5120, 6144],
                        help='Memory size for serverless (MB)')
    parser.add_argument('--max-concurrency', type=int, default=10, 
                        help='Max concurrent invocations for serverless (1-200)')
    
    args = parser.parse_args()
    
    # Create temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        # Download model from MLflow
        model_path = download_mlflow_model(
            args.model_name,
            args.model_version,
            args.model_stage,
            args.mlflow_uri,
            tmpdir
        )
        
        # Create model archive
        archive_path = os.path.join(tmpdir, "model.tar.gz")
        inference_script = os.path.join(os.path.dirname(__file__), "sagemaker_inference.py")
        
        create_model_archive(model_path, inference_script, archive_path)
        
        # Upload to S3
        timestamp = pd.Timestamp.now().strftime('%Y%m%d-%H%M%S')
        s3_key = f"{args.s3_prefix}/{args.model_name}/v{args.model_version}/{timestamp}/model.tar.gz"
        model_data_url = upload_to_s3(archive_path, args.s3_bucket, s3_key)
        
        # Create SageMaker model — include timestamp so the name is always unique
        # and never collides with a stale model from a previous failed deploy.
        sm_model_name = f"{args.model_name}-v{args.model_version}-{timestamp}".replace('_', '-')

        create_sagemaker_model(sm_model_name, model_data_url, args.role, args.region)
        
        # Create endpoint configuration
        config_name = f"{sm_model_name}-cfg"
        create_endpoint_config(
            config_name, 
            sm_model_name, 
            args.instance_type, 
            args.region,
            serverless=args.serverless,
            memory_size_mb=args.memory_size_mb,
            max_concurrency=args.max_concurrency
        )
        
        # Create or update endpoint
        create_or_update_endpoint(args.endpoint_name, config_name, args.region)
        
        print("\n" + "="*60)
        print("Deployment complete!")
        print("="*60)
        print(f"Endpoint name: {args.endpoint_name}")
        print(f"Region: {args.region}")
        print(f"\nTest the endpoint:")
        print(f"aws sagemaker-runtime invoke-endpoint \\")
        print(f"  --endpoint-name {args.endpoint_name} \\")
        print(f"  --body '{{\"texts\": [\"I feel great today!\"]}}' \\")
        print(f"  --content-type application/json \\")
        print(f"  --region {args.region} \\")
        print(f"  output.json")


if __name__ == "__main__":
    main()
