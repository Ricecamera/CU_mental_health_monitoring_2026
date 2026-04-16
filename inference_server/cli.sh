# ── Logistic Regression / TF-IDF baseline ────────────────────────────────────
# python deploy_to_sagemaker.py \
# --model-name mental-health-logreg-tfidf \
# --model-version 7 \
# --serverless

# ── BERT model3 v2  (serverless, PyTorch DLC) ────────────────────────────────
python deploy_to_sagemaker.py \
--model-name distilbert-mental-health \
--model-version 6 \
--serverless \
--memory-size-mb 3072 \
--container-image 763104351884.dkr.ecr.ap-southeast-1.amazonaws.com/pytorch-inference:2.1.0-cpu-py310-ubuntu20.04-sagemaker \
--requirements-file sagemaker_requirements_bert.txt

# ── Tail endpoint logs ────────────────────────────────────────────────────────
# aws logs describe-log-streams --log-group-name "/aws/sagemaker/Endpoints/mental-health-inference" --region ap-southeast-1 --order-by LastEventTime --descending --max-items 5 --no-cli-pager
# aws logs get-log-events --log-group-name "/aws/sagemaker/Endpoints/mental-health-inference" --log-stream-name "<log-stream-name>" --region ap-southeast-1 --no-cli-pager --output text