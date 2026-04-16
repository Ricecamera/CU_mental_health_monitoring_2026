# Mental Health Inference Server

Deploy mental health text classification models from MLflow to AWS SageMaker (Real-Time or Serverless).

## Features

- 🚀 One-command deployment from MLflow to SageMaker
- 📊 Real-Time or Serverless endpoint options
- 🔄 Zero-downtime model updates
- 💰 Flexible pricing (hourly or pay-per-request)

## Prerequisites

### 1. AWS CLI Setup

```bash
aws configure
```

Enter your AWS credentials and region (ap-southeast-1).

### 2. Create IAM Role for SageMaker

**In AWS Console:**
1. Go to **IAM** → **Roles** → **Create role**
2. Select **AWS service** → **SageMaker**
3. Attach policies: `AmazonSageMakerFullAccess`
4. Add inline policy for S3 access:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["s3:GetObject", "s3:PutObject", "s3:DeleteObject", "s3:ListBucket"],
      "Resource": [
        "arn:aws:s3:::guessing-engineering-sagemaker-models",
        "arn:aws:s3:::guessing-engineering-sagemaker-models/*"
      ]
    },
    {
      "Sid": "AllowSageMakerECR",
      "Effect": "Allow",
      "Action": [
        "ecr:GetAuthorizationToken",
        "ecr:BatchCheckLayerAvailability",
        "ecr:GetDownloadUrlForLayer",
        "ecr:BatchGetImage"
      ],
      "Resource": "*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "logs:CreateLogGroup",
        "logs:CreateLogStream",
        "logs:PutLogEvents"
      ],
      "Resource": "arn:aws:logs:*:*:*"
    }
  ]
}
```

5. Name the role: `SageMakerExecutionRole`
6. Copy the role ARN (e.g., `arn:aws:iam::123456789012:role/SageMakerExecutionRole`)

### 3. Create S3 Bucket

```bash
aws s3 mb s3://guessing-engineering-sagemaker-models --region ap-southeast-1
```

### 4. Install Python Dependencies

```bash
pip install boto3 mlflow sagemaker pandas
```

### 5. Verify IAM Permissions

Your AWS CLI user needs permissions to deploy. See **[IAM_PERMISSIONS.md](IAM_PERMISSIONS.md)** for the complete policy.

**Quick test:**
```bash
aws sts get-caller-identity
aws s3 ls s3://guessing-engineering-sagemaker-models
aws sagemaker list-models --region ap-southeast-1
```

## Deploy to SageMaker

### Option 1: Serverless (Recommended for Development/Low Traffic)

```bash
python deploy_to_sagemaker.py \
  --mlflow-uri http://ec2-18-136-206-38.ap-southeast-1.compute.amazonaws.com/ \
  --model-name mental-health-logreg-tfidf \
  --model-version 1 \
  --role arn:aws:iam::YOUR_ACCOUNT:role/SageMakerExecutionRole \
  --s3-bucket guessing-engineering-sagemaker-models \
  --endpoint-name mental-health-inference \
  --region ap-southeast-1 \
  --serverless \
  --memory-size-mb 2048 \
  --max-concurrency 10
```

**Cost**: ~$5/month for 10K requests/day

### Option 2: Real-Time (For Production/High Traffic)

```bash
python deploy_to_sagemaker.py \
  --mlflow-uri http://ec2-18-136-206-38.ap-southeast-1.compute.amazonaws.com/ \
  --model-name mental-health-logreg-tfidf \
  --model-version 1 \
  --role arn:aws:iam::YOUR_ACCOUNT:role/SageMakerExecutionRole \
  --s3-bucket guessing-engineering-sagemaker-models \
  --endpoint-name mental-health-inference \
  --region ap-southeast-1 \
  --instance-type ml.t2.medium
```

**Cost**: ~$47/month (24/7)

**Common Instance Types:**
- `ml.t2.medium` - $0.065/hour (Dev/Testing)
- `ml.m5.large` - $0.134/hour (Production)
- `ml.m5.xlarge` - $0.269/hour (High traffic)

### Key Parameters

- `--mlflow-uri`: Your MLflow server URL
- `--model-name`: Model name in MLflow registry
- `--model-version`: Version number (or use `--model-stage Production`)
- `--role`: SageMaker execution role ARN (from step 2)
- `--s3-bucket`: S3 bucket name (from step 3)
- `--endpoint-name`: Choose a name for your endpoint
- `--region`: AWS region

**Deployment takes 5-10 minutes.**

## Test the Endpoint

```bash
# Using test script
python test_sagemaker_endpoint.py \
  --endpoint-name mental-health-inference \
  --region ap-southeast-1

# Or using AWS CLI
aws sagemaker-runtime invoke-endpoint \
  --endpoint-name mental-health-inference \
  --body '{"texts": ["I feel great today!"]}' \
  --content-type application/json \
  --region ap-southeast-1 \
  output.json
```

## Update Model

Deploy a new version (zero downtime):

```bash
python deploy_to_sagemaker.py \
  --model-version 2 \
  --endpoint-name mental-health-inference \
  # ... same other parameters
```

## Cleanup (Stop Billing)

```bash
# Delete endpoint
aws sagemaker delete-endpoint \
  --endpoint-name mental-health-inference \
  --region ap-southeast-1

# Recreate later using same config
aws sagemaker list-endpoint-configs --region ap-southeast-1
aws sagemaker create-endpoint \
  --endpoint-name mental-health-inference \
  --endpoint-config-name <CONFIG_NAME>
```

## Monitoring & Troubleshooting

**View logs:**
```bash
aws logs tail /aws/sagemaker/Endpoints/mental-health-inference --follow
```

**Common issues:**
- IAM role missing permissions → See [IAM_PERMISSIONS.md](IAM_PERMISSIONS.md)
- Endpoint creation fails → Check CloudWatch logs
- High costs → Use serverless or delete endpoint when not in use


### Request/Response Example

Request:

```json
{"texts": ["I feel overwhelmed lately"]}
```

Response:

```json
{
  "predictions": [
    {
      "text": "I feel overwhelmed lately",
      "predicted_label": "Negative",
      "confidence": 0.82
    }
  ]
}
```

## Additional Resources

- **[IAM_PERMISSIONS.md](IAM_PERMISSIONS.md)** - Complete AWS IAM permissions guide
- [AWS SageMaker Documentation](https://docs.aws.amazon.com/sagemaker/)
- [How SageMaker uses your inference code (`model_fn`, `input_fn`, `predict_fn`, `output_fn`)](https://aws.amazon.com/blogs/machine-learning/create-a-sagemaker-inference-endpoint-with-custom-model-extended-container/#:~:text=In%20such%20cases%2C%20SageMaker%20allows,specific%20use%20case%20and%20requirements.)
- [SageMaker Inference Toolkit default handler service (reference implementation)](https://github.com/aws/sagemaker-inference-toolkit/blob/master/src/sagemaker_inference/default_handler_service.py)