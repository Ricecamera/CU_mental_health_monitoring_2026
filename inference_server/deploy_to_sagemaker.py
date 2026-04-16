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
import argparse
import tarfile
import tempfile
import pandas as pd

import boto3
import botocore.exceptions
import mlflow

# ── Project defaults ──────────────────────────────────────────────────────────
MLFLOW_URI      = "http://ec2-18-136-206-38.ap-southeast-1.compute.amazonaws.com/"
ROLE_ARN        = "arn:aws:iam::040533191971:role/SageMakerExecutionRole"
S3_BUCKET       = "guessing-engineering-sagemaker-models"
S3_PREFIX       = "sagemaker/mental-health-models"
REGION          = "ap-southeast-1"
ENDPOINT_NAME   = "mental-health-inference"
ECR_ACCOUNT     = "121021644041"
# ─────────────────────────────────────────────────────────────────────────────

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


def create_model_archive(model_path, inference_script, output_path, requirements_file=None):
    if requirements_file is None:
        requirements_file = os.path.join(os.path.dirname(inference_script), "sagemaker_requirements.txt")

    with tarfile.open(output_path, "w:gz") as tar:
        for item in os.listdir(model_path):
            if item != "requirements.txt":
                tar.add(os.path.join(model_path, item), arcname=item)

        tar.add(inference_script, arcname="code/inference.py")

        setup_py = b"from setuptools import setup\nsetup(name='inference', version='1.0.0', py_modules=['inference'])\n"
        info = tarfile.TarInfo(name="code/setup.py")
        info.size = len(setup_py)
        tar.addfile(info, io.BytesIO(setup_py))

        if os.path.exists(requirements_file):
            tar.add(requirements_file, arcname="code/requirements.txt")

    print(f"Archive created: {output_path}")


def upload_to_s3(file_path, bucket, key):
    """Upload model archive to S3"""
    print(f"Uploading to s3://{bucket}/{key}")
    s3_client = boto3.client('s3')
    s3_client.upload_file(file_path, bucket, key)
    print("Upload complete")
    return f"s3://{bucket}/{key}"


def create_sagemaker_model(model_name, model_data_url, image_uri=None):
    sm = boto3.client('sagemaker', region_name=REGION)
    if image_uri is None:
        image_uri = f"{ECR_ACCOUNT}.dkr.ecr.{REGION}.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3"
    try:
        sm.delete_model(ModelName=model_name)
    except Exception:
        pass
    response = sm.create_model(
        ModelName=model_name,
        PrimaryContainer={
            'Image': image_uri,
            'ModelDataUrl': model_data_url,
            'Environment': {
                'SAGEMAKER_PROGRAM': 'inference.py',
                'SAGEMAKER_SUBMIT_DIRECTORY': '/opt/ml/model/code',
            }
        },
        ExecutionRoleArn=ROLE_ARN
    )
    print(f"Model created: {response['ModelArn']}")
    return response['ModelArn']


def create_endpoint_config(config_name, model_name, serverless=False, memory_size_mb=2048, max_concurrency=10, instance_type='ml.t2.medium'):
    sm = boto3.client('sagemaker', region_name=REGION)
    variant = {'VariantName': 'AllTraffic', 'ModelName': model_name, 'InitialVariantWeight': 1.0}
    if serverless:
        
        variant['ServerlessConfig'] = {'MemorySizeInMB': memory_size_mb, 'MaxConcurrency': max_concurrency}
    else:
        variant.update({'InitialInstanceCount': 1, 'InstanceType': instance_type})
    try:
        response = sm.create_endpoint_config(
            EndpointConfigName=config_name,
            ProductionVariants=[variant]
        )
        print(f"Endpoint config created: {response['EndpointConfigArn']}")
        return response['EndpointConfigArn']
    except sm.exceptions.ResourceInUse:
        print(f"Endpoint config {config_name} already exists")
        return None


def create_or_update_endpoint(endpoint_name, config_name):
    sm = boto3.client('sagemaker', region_name=REGION)
    try:
        status = sm.describe_endpoint(EndpointName=endpoint_name)['EndpointStatus']
        if status == 'Failed':
            print(f"Endpoint in FAILED state — deleting: {endpoint_name}")
            sm.delete_endpoint(EndpointName=endpoint_name)
            sm.get_waiter('endpoint_deleted').wait(EndpointName=endpoint_name)
            print(f"Creating endpoint: {endpoint_name}")
            sm.create_endpoint(EndpointName=endpoint_name, EndpointConfigName=config_name)
        else:
            sm.update_endpoint(EndpointName=endpoint_name, EndpointConfigName=config_name)
            print(f"Updating endpoint: {endpoint_name}")
    except botocore.exceptions.ClientError:
        sm.create_endpoint(EndpointName=endpoint_name, EndpointConfigName=config_name)
        print(f"Creating endpoint: {endpoint_name}")
    sm.get_waiter('endpoint_in_service').wait(EndpointName=endpoint_name)
    print(f"Endpoint {endpoint_name} is in service!")


def main():
    parser = argparse.ArgumentParser(description='Deploy MLflow model to SageMaker')
    parser.add_argument('--mlflow-uri', default=MLFLOW_URI)
    parser.add_argument('--model-name', required=True)
    parser.add_argument('--model-version', default='1')
    parser.add_argument('--model-stage', default=None)
    parser.add_argument('--region', default=REGION)
    parser.add_argument('--role', default=ROLE_ARN)
    parser.add_argument('--s3-bucket', default=S3_BUCKET)
    parser.add_argument('--s3-prefix', default=S3_PREFIX)
    parser.add_argument('--endpoint-name', default=ENDPOINT_NAME)
    parser.add_argument('--instance-type', default='ml.t2.medium')
    parser.add_argument('--serverless', action='store_true')
    parser.add_argument('--memory-size-mb', type=int, default=2048,
                        choices=[1024, 2048, 3072, 4096, 5120, 6144])
    parser.add_argument('--max-concurrency', type=int, default=10)
    parser.add_argument('--container-image', default=None)
    parser.add_argument('--requirements-file', default=None)
    
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
        
        create_model_archive(model_path, inference_script, archive_path, requirements_file=args.requirements_file)
        
        # Upload to S3
        timestamp = pd.Timestamp.now().strftime('%Y%m%d-%H%M%S')
        s3_key = f"{args.s3_prefix}/{args.model_name}/v{args.model_version}/{timestamp}/model.tar.gz"
        model_data_url = upload_to_s3(archive_path, args.s3_bucket, s3_key)
        
        # Create SageMaker model — include timestamp so the name is always unique
        # and never collides with a stale model from a previous failed deploy.
        sm_model_name = f"{args.model_name}-v{args.model_version}-{timestamp}".replace('_', '-')

        create_sagemaker_model(sm_model_name, model_data_url, image_uri=args.container_image)

        config_name = f"{sm_model_name}-cfg"
        create_endpoint_config(
            config_name,
            sm_model_name,
            serverless=args.serverless,
            memory_size_mb=args.memory_size_mb,
            max_concurrency=args.max_concurrency,
            instance_type=args.instance_type,
        )

        create_or_update_endpoint(args.endpoint_name, config_name)
        
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
