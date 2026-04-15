"""
SageMaker inference script for mental health text classification.
This script implements the required SageMaker inference handlers.
"""
import os
import re
import json
import logging
from io import StringIO

import pandas as pd
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Label mapping
LABEL2ID = {
    "Normal": 0,
    "Negative": 1,
    "Very Negative": 2,
    "Suicidal": 3,
}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


def clean_text(text: str) -> str:
    """
    Clean and preprocess text for model input.
    Same preprocessing as training pipeline.
    """
    text = text.lower()
    text = text.encode("ascii", "ignore").decode()
    text = re.sub(r'[^a-z0-9\s!?]', '', text)
    return text


def model_fn(model_dir):
    """
    Load the model from the model directory.
    This function is called once when the inference container starts.
    
    SageMaker will download the model.tar.gz and extract it to model_dir.
    
    Args:
        model_dir: Directory where model artifacts are stored
        
    Returns:
        Loaded model object
    """
    logger.info(f"Loading model from {model_dir}")
    
    try:
        # MLflow saves sklearn models as model.pkl inside a 'model' subdirectory
        model_pkl = os.path.join(model_dir, "model", "model.pkl")
        
        if os.path.exists(model_pkl):
            logger.info(f"Loading model from MLflow format: {model_pkl}")
            model = joblib.load(model_pkl)
        else:
            # Fallback: try direct model.pkl in root
            model_file = os.path.join(model_dir, "model.pkl")
            logger.info(f"Loading model from: {model_file}")
            model = joblib.load(model_file)
        
        logger.info("Model loaded successfully")
        return model
    
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise


def input_fn(request_body, request_content_type):
    """
    Deserialize and prepare the input data.
    
    SageMaker calls this function to convert the request payload into
    a format that can be consumed by predict_fn.
    
    Args:
        request_body: The request payload
        request_content_type: The content type of the request
        
    Returns:
        Deserialized input data (list of texts)
    """
    logger.info(f"Processing input with content type: {request_content_type}")
    
    if request_content_type == "application/json":
        data = json.loads(request_body)
        
        # Support both single text and list of texts
        if isinstance(data, dict):
            if "texts" in data:
                texts = data["texts"]
            elif "text" in data:
                texts = [data["text"]]
            elif "instances" in data:
                # Support SageMaker's default format
                texts = [inst.get("text", inst) if isinstance(inst, dict) else inst 
                        for inst in data["instances"]]
            else:
                raise ValueError("Request must contain 'text', 'texts', or 'instances' field")
        elif isinstance(data, list):
            texts = data
        elif isinstance(data, str):
            texts = [data]
        else:
            raise ValueError(f"Unsupported data format: {type(data)}")
        
        return texts
    
    elif request_content_type == "text/plain":
        # Single text input
        return [request_body.decode("utf-8")]
    
    elif request_content_type == "text/csv":
        # CSV format with 'text' column
        df = pd.read_csv(StringIO(request_body.decode("utf-8")))
        if "text" not in df.columns:
            raise ValueError("CSV must contain 'text' column")
        return df["text"].tolist()
    
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")


def predict_fn(input_data, model):
    """
    Make predictions on the deserialized data.
    
    Args:
        input_data: List of text strings from input_fn
        model: Loaded model from model_fn
        
    Returns:
        Predictions dictionary
    """
    logger.info(f"Making predictions for {len(input_data)} samples")
    
    try:
        # Preprocess texts
        cleaned_texts = [clean_text(text) for text in input_data]
        
        # Create DataFrame for sklearn pipeline
        df = pd.DataFrame({"text": cleaned_texts})
        
        # Get predictions and probabilities
        predictions = model.predict(df["text"])
        probabilities = model.predict_proba(df["text"])
        
        # Format results
        results = []
        for i, (pred, probs) in enumerate(zip(predictions, probabilities)):
            label = ID2LABEL.get(int(pred), f"Unknown_{pred}")
            
            # Create probability dict for all classes
            prob_dict = {
                ID2LABEL.get(j, f"Class_{j}"): float(prob)
                for j, prob in enumerate(probs)
            }
            
            results.append({
                "text": input_data[i],
                "predicted_label": label,
                "predicted_class_id": int(pred),
                "confidence": float(max(probs)),
                "probabilities": prob_dict
            })
        
        logger.info("Predictions completed successfully")
        return {"predictions": results}
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise


def output_fn(prediction, response_content_type):
    """
    Serialize the prediction result.
    
    Args:
        prediction: Output from predict_fn
        response_content_type: Desired response content type
        
    Returns:
        Serialized prediction
    """
    logger.info(f"Serializing output as {response_content_type}")
    
    if response_content_type == "application/json":
        return json.dumps(prediction)
    
    elif response_content_type == "text/csv":
        # Convert to CSV format
        df = pd.DataFrame(prediction["predictions"])
        return df.to_csv(index=False)
    
    else:
        raise ValueError(f"Unsupported response content type: {response_content_type}")
