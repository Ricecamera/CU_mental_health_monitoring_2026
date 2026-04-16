# ── Logistic Regression / TF-IDF baseline ────────────────────────────────────
# python deploy_to_sagemaker.py \
# --model-name mental-health-logreg-tfidf \
# --model-version 7 \
# --serverless

# ── BERT model3 v2  (serverless, PyTorch DLC) ────────────────────────────────
# python deploy_to_sagemaker.py \
# --model-name distilbert-mental-health \
# --model-version 6 \
# --serverless \
# --memory-size-mb 3072 \
# --container-image 763104351884.dkr.ecr.ap-southeast-1.amazonaws.com/pytorch-inference:2.1.0-cpu-py310-ubuntu20.04-sagemaker \
# --requirements-file sagemaker_requirements_bert.txt

# ── Tail endpoint logs ────────────────────────────────────────────────────────
# MSYS_NO_PATHCONV=1 aws logs get-log-events --log-group-name /aws/sagemaker/Endpoints/mental-health-inference --log-stream-name "AllTraffic/c9492bd6300af587abaa038eda7dff03-95d8b938a35e4149925839a1abc23432" --region ap-southeast-1 --no-cli-pager | python -c "import sys,json; [print(e['message']) for e in json.load(sys.stdin).get('events',[])]"