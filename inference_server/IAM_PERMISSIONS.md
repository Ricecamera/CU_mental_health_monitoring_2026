# AWS IAM Permissions Guide

## Required IAM Permissions for deploy_to_sagemaker.py

The AWS CLI user running `deploy_to_sagemaker.py` needs the following permissions:

## Complete IAM Policy (JSON)

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "SageMakerModelDeployment",
            "Effect": "Allow",
            "Action": [
                "sagemaker:CreateModel",
                "sagemaker:CreateEndpointConfig",
                "sagemaker:CreateEndpoint",
                "sagemaker:UpdateEndpoint",
                "sagemaker:DescribeEndpoint",
                "sagemaker:DescribeEndpointConfig",
                "sagemaker:DescribeModel",
                "sagemaker:ListModels",
                "sagemaker:ListTags",
                "sagemaker:DeleteEndpoint",
                "sagemaker:DeleteEndpointConfig",
                "sagemaker:DeleteModel",
                "sagemaker:InvokeEndpoint",
            ],
            "Resource": [
                "arn:aws:sagemaker:ap-southeast-1:*:model/*",
                "arn:aws:sagemaker:ap-southeast-1:*:endpoint-config/*",
                "arn:aws:sagemaker:ap-southeast-1:*:endpoint/*"
            ]
        },
        {
            "Sid": "S3ModelArtifacts",
            "Effect": "Allow",
            "Action": [
                "s3:PutObject",
                "s3:GetObject",
                "s3:ListBucket"
            ],
            "Resource": [
                "arn:aws:s3:::guessing-engineering-sagemaker-models",
                "arn:aws:s3:::guessing-engineering-sagemaker-models/*",
                "arn:aws:s3:::cu-guessing-engineering-dvc",
                "arn:aws:s3:::cu-guessing-engineering-dvc/*"
            ]
        },
        {
            "Sid": "S3MLflowArtifacts",
            "Effect": "Allow",
            "Action": [
                "s3:GetObject",
                "s3:ListBucket"
            ],
            "Resource": [
                "arn:aws:s3:::cu-guessing-engineering-dvc",
                "arn:aws:s3:::cu-guessing-engineering-dvc/*"
            ]
        },
        {
            "Sid": "PassRoleToSageMaker",
            "Effect": "Allow",
            "Action": "iam:PassRole",
            "Resource": "arn:aws:iam::*:role/SageMakerExecutionRole",
            "Condition": {
                "StringEquals": {
                    "iam:PassedToService": "sagemaker.amazonaws.com"
                }
            }
        },
        {
            "Sid": "GetCallerIdentity",
            "Effect": "Allow",
            "Action": "sts:GetCallerIdentity",
            "Resource": "*"
        },
        {
            "Sid": "ECRContainerAccess",
            "Effect": "Allow",
            "Action": [
                "ecr:GetAuthorizationToken",
                "ecr:BatchCheckLayerAvailability",
                "ecr:GetDownloadUrlForLayer",
                "ecr:BatchGetImage"
            ],
            "Resource": "*"
        }
    ]
}
```

## Creating IAM User with Permissions

### Option 1: AWS Console

1. Go to **IAM Console** → **Users** → **Add users**
2. Enter username (e.g., `sagemaker-deployer`)
3. Select **Access key - Programmatic access**
4. Click **Next: Permissions**
5. Select **Attach policies directly**
6. Click **Create policy** → **JSON** tab
7. Paste the complete policy above (replace bucket name)
8. Name it `SageMakerDeploymentPolicy`
9. Go back and attach this policy to the user
10. Click **Next** → **Create user**
11. **Save Access Key ID and Secret Access Key**

## Configure AWS CLI

After creating the user and getting access keys:

```bash
aws configure
```

Enter:
- **AWS Access Key ID**: From user creation
- **AWS Secret Access Key**: From user creation
- **Default region**: `ap-southeast-1`
- **Default output format**: `json`

## Verify Permissions

Test that your credentials work:

```bash
# 1. Check identity
aws sts get-caller-identity

# 2. Check S3 access
aws s3 ls s3://guessing-engineering-sagemaker-models

# 3. Check SageMaker access
aws sagemaker list-models --region ap-southeast-1
```

## Troubleshooting Permission Issues

### Error: "User is not authorized to perform..."

**Cause**: Missing IAM permission

**Solution**: Check which action failed and add it to the policy:
```bash
# Common issues:
sagemaker:CreateModel          # Can't create model
sagemaker:CreateEndpoint       # Can't create endpoint
s3:PutObject                   # Can't upload to S3
iam:PassRole                   # Can't pass role to SageMaker
```

### Error: "Access Denied" on S3

**Cause**: Bucket name in policy doesn't match actual bucket

**Solution**: Update policy with correct bucket name:
```bash
# Check your bucket name
aws s3 ls

# Update policy Resource field
"Resource": [
  "arn:aws:s3:::your-actual-bucket-name",
  "arn:aws:s3:::your-actual-bucket-name/*"
]
```

### Error: "Cannot assume role"

**Cause**: IAM user can't pass the SageMaker execution role

**Solution**: 
1. Verify the role ARN is correct
2. Ensure PassRole permission includes this role:
```json
"Resource": "arn:aws:iam::YOUR_ACCOUNT:role/SageMakerExecutionRole"
```

### Error: "Invalid endpoint configuration"

**Cause**: Region mismatch between resources

**Solution**: Ensure all resources are in the same region:
```bash
# Check region
aws configure get region

# Or override in command
python deploy_to_sagemaker.py --region ap-southeast-1 ...
```

## Minimum Permissions (Development Only)

For quick testing in a development account, you can use:
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "sagemaker:*",
        "s3:*",
        "iam:PassRole",
        "sts:GetCallerIdentity",
        "ecr:*"
      ],
      "Resource": "*"
    }
  ]
}
```

**⚠️ WARNING**: This is overly permissive. Only use in sandbox/dev environments!

## Additional Cleanup Permissions

If you want to delete resources, add:
```json
{
  "Sid": "SageMakerCleanup",
  "Effect": "Allow",
  "Action": [
    "sagemaker:DeleteEndpoint",
    "sagemaker:DeleteEndpointConfig",
    "sagemaker:DeleteModel"
  ],
  "Resource": [
    "arn:aws:sagemaker:ap-southeast-1:*:endpoint/*",
    "arn:aws:sagemaker:ap-southeast-1:*:endpoint-config/*",
    "arn:aws:sagemaker:ap-southeast-1:*:model/*"
  ]
}
```

## Summary Checklist

Before running `deploy_to_sagemaker.py`:

- [ ] IAM user created with programmatic access
- [ ] SageMakerDeploymentPolicy attached to user
- [ ] S3 bucket name updated in policy
- [ ] SageMaker execution role ARN noted
- [ ] AWS CLI configured with access keys
- [ ] `aws sts get-caller-identity` works
- [ ] `aws s3 ls s3://your-bucket` works
- [ ] Region set to `ap-southeast-1`