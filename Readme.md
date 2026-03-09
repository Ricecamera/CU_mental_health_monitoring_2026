# CU Mental Health Monitoring

## Description

โปรเจคส่งวิชา Software Engineering for Machine Learning Systems 2110555
ปีการศึกษา 2025/2 จุฬาลงกรณ์มหาวิทยาลัย

## Group Name

Guessing Engineering

## Group Members
- 6530112221 Napatpong Limpasonthipong
- 6872093121 Sahatsarin Pawanna
- 6872035821 Tanabodee Yambangyang
- 6870190021 Potsawat Monchaiyapoom
- 6633269821 Suradet Tongjaijongjaroen

## Dataset Setup

This project uses DVC with Amazon S3 for dataset versioning and remote storage.

### 1. Install AWS CLI

Install AWS CLI on your machine before configuring credentials.

### 2. Install DVC

```bash
pip install "dvc"
pip install "dvc[s3]"
dvc --version
```

### 3. Configure AWS Credentials

Option 1: configure credentials interactively

```bash
aws configure
```

Option 2: set environment variables manually (for window)

```powershell
$env:AWS_ACCESS_KEY_ID="..."
$env:AWS_SECRET_ACCESS_KEY="..."
$env:AWS_DEFAULT_REGION="ap-southeast-1"
```

### 4. Pull the Dataset

```bash
dvc pull
```

### 5. Push a New Dataset Version

```bash
dvc push
```