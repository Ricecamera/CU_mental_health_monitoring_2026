"""
SageMaker inference script for DistilBERT mental health text classification.
Model : model3  Version : 2
Architecture : distilbert-base-uncased fine-tuned, 4-class classifier
Labels : 0=Normal  1=Anxiety  2=Depression  3=Suicidal
"""
import os
import re
import json
import string
import logging
from turtle import pd

import nltk
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ID2LABEL = {0: "Normal", 1: "Anxiety", 2: "Depression", 3: "Suicidal"}
BERT_MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 128

_stop_words = set(stopwords.words("english"))
_negate_raw = list(filter(lambda x: x.endswith(("'t", "'nt")), stopwords.words("english"))) + ["can't"]
_negate_aux = list(map(lambda x: x.replace("'", ""), _negate_raw)) + list(map(lambda x: x[:-2], _negate_raw))
_lemmatizer = WordNetLemmatizer()


def clean_text(text):
    text = text.lower()
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = text.replace("'", "")
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = " ".join(["not" if w in _negate_aux else w for w in text.split()])
    text = " ".join([w for w in text.split() if w not in _stop_words or w == "not"])
    text = re.sub(r"\d+", "", text)
    text = " ".join([_lemmatizer.lemmatize(w) for w in text.split()])
    return text


def model_fn(model_dir):
    logger.info(f"Loading BERT model from {model_dir}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AutoModelForSequenceClassification.from_pretrained(BERT_MODEL_NAME, num_labels=4)

    candidates = [
        os.path.join(model_dir, "best_model.pth"),
        os.path.join(model_dir, "data", "model.pth"),
        os.path.join(model_dir, "model.pth"),
    ]
    weights_path = next((p for p in candidates if os.path.exists(p)), None)
    if not weights_path:
        raise FileNotFoundError(f"No weights found in {model_dir}. Tried: {candidates}")

    logger.info(f"Loading weights from {weights_path}")
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
    logger.info("Model and tokenizer ready")
    return {"model": model, "tokenizer": tokenizer, "device": device}


def input_fn(request_body, request_content_type):
    logger.info(f"Deserializing input — content-type: {request_content_type}")

    if request_content_type == "application/json":
        data = json.loads(request_body)
        if isinstance(data, dict):
            if "texts" in data:
                return data["texts"]
            if "text" in data:
                return [data["text"]]
            if "instances" in data:
                return [
                    inst.get("text", inst) if isinstance(inst, dict) else inst
                    for inst in data["instances"]
                ]
            raise ValueError("JSON body must contain 'text', 'texts', or 'instances'")
        if isinstance(data, list):
            return data
        if isinstance(data, str):
            return [data]
        raise ValueError(f"Unsupported JSON data type: {type(data)}")

    if request_content_type == "text/plain":
        body = request_body if isinstance(request_body, str) else request_body.decode("utf-8")
        return [body]

    raise ValueError(f"Unsupported content type: {request_content_type}")


def predict_fn(input_data, model_artifacts):
    model = model_artifacts["model"]
    tokenizer = model_artifacts["tokenizer"]
    device = model_artifacts["device"]

    logger.info(f"Running inference on {len(input_data)} sample(s)")

    try: 
        cleaned = [clean_text(t) for t in input_data]
        encoding = tokenizer(
            cleaned,
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )

        with torch.no_grad():
            outputs = model(
                encoding["input_ids"].to(device),
                attention_mask=encoding["attention_mask"].to(device),
            )
            probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()

        preds = probs.argmax(axis=1)
        results = []
        for i, (pred, prob_row) in enumerate(zip(preds, probs)):
            label = ID2LABEL.get(int(pred))
            # Create probability dict for all classes
            prob_dict = {
                ID2LABEL.get(j, f"Class_{j}"): float(prob)
                for j, prob in enumerate(prob_row)
            }

            results.append({
                "text": input_data[i],
                "predicted_label": label,
                "predicted_class_id": int(pred),
                "confidence": float(prob_row[pred]),
                "probabilities": prob_dict,
            }
            )
        logger.info("Predictions completed successfully")
        return {
            "predictions": results,
            "device": str(device),
            "cuda_available": torch.cuda.is_available(),
        }
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise


def output_fn(prediction, response_content_type):
    logger.info(f"Serializing output as {response_content_type}")
    if response_content_type == "application/json":
        return json.dumps(prediction)
    elif response_content_type == "text/csv":
        # Convert to CSV format
        df = pd.DataFrame(prediction["predictions"])
        return df.to_csv(index=False)

    else:
        raise ValueError(f"Unsupported response content type: {response_content_type}")

