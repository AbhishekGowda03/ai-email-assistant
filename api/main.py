from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import numpy as np
from typing import Dict, List
import time

app = FastAPI(title="AI Email Assistant API", version="1.0")

# Add CORS middleware - ADD THIS SECTION
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models at startup
print("Loading models...")

# Load SVM (best overall)
svm_model = joblib.load('../models/baseline/linear_svm.pkl')
vectorizer = joblib.load('../models/baseline/tfidf_vectorizer.pkl')

# Load BERT (optional, for comparison)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
bert_tokenizer = BertTokenizer.from_pretrained('../models/transformer/')
bert_model = BertForSequenceClassification.from_pretrained('../models/transformer/').to(device)
bert_model.eval()

print("✓ Models loaded successfully!")

class EmailRequest(BaseModel):
    text: str
    model_type: str = "svm"  # "svm", "bert", or "all"

class EmailResponse(BaseModel):
    is_spam: bool
    confidence: float
    prediction: str
    model_used: str
    latency_ms: float

class BatchEmailRequest(BaseModel):
    emails: List[str]
    model_type: str = "svm"

@app.get("/")
def root():
    """API health check"""
    return {
        "message": "AI Email Assistant API",
        "status": "running",
        "models": ["svm", "bert"],
        "endpoints": ["/classify", "/classify_batch", "/models/info"]
    }

@app.post("/classify", response_model=EmailResponse)
def classify_email(request: EmailRequest):
    """Classify a single email"""
    
    start_time = time.time()
    
    try:
        if request.model_type == "svm":
            # SVM prediction
            X = vectorizer.transform([request.text])
            prediction = svm_model.predict(X)[0]
            # Get decision function for confidence
            decision = svm_model.decision_function(X)[0]
            confidence = 1 / (1 + np.exp(-decision))  # Sigmoid
            
        elif request.model_type == "bert":
            # BERT prediction
            encoding = bert_tokenizer(
                request.text,
                add_special_tokens=True,
                max_length=128,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )
            
            with torch.no_grad():
                input_ids = encoding['input_ids'].to(device)
                attention_mask = encoding['attention_mask'].to(device)
                outputs = bert_model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=1)
                prediction = torch.argmax(logits, dim=1).item()
                confidence = probabilities[0][prediction].item()
        
        else:
            raise HTTPException(status_code=400, detail="Invalid model_type. Use 'svm' or 'bert'")
        
        latency = (time.time() - start_time) * 1000  # Convert to ms
        
        return EmailResponse(
            is_spam=bool(prediction),
            confidence=float(confidence),
            prediction="Spam" if prediction == 1 else "Ham",
            model_used=request.model_type.upper(),
            latency_ms=round(latency, 2)
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/classify_batch")
def classify_batch(request: BatchEmailRequest):
    """Classify multiple emails"""
    
    start_time = time.time()
    results = []
    
    try:
        if request.model_type == "svm":
            X = vectorizer.transform(request.emails)
            predictions = svm_model.predict(X)
            decisions = svm_model.decision_function(X)
            confidences = 1 / (1 + np.exp(-decisions))
            
            for email, pred, conf in zip(request.emails, predictions, confidences):
                results.append({
                    "email": email[:100] + "..." if len(email) > 100 else email,
                    "is_spam": bool(pred),
                    "confidence": float(conf),
                    "prediction": "Spam" if pred == 1 else "Ham"
                })
        
        elif request.model_type == "bert":
            for email in request.emails:
                encoding = bert_tokenizer(
                    email,
                    add_special_tokens=True,
                    max_length=128,
                    padding='max_length',
                    truncation=True,
                    return_attention_mask=True,
                    return_tensors='pt'
                )
                
                with torch.no_grad():
                    input_ids = encoding['input_ids'].to(device)
                    attention_mask = encoding['attention_mask'].to(device)
                    outputs = bert_model(input_ids=input_ids, attention_mask=attention_mask)
                    logits = outputs.logits
                    probabilities = torch.softmax(logits, dim=1)
                    prediction = torch.argmax(logits, dim=1).item()
                    confidence = probabilities[0][prediction].item()
                
                results.append({
                    "email": email[:100] + "..." if len(email) > 100 else email,
                    "is_spam": bool(prediction),
                    "confidence": float(confidence),
                    "prediction": "Spam" if prediction == 1 else "Ham"
                })
        
        latency = (time.time() - start_time) * 1000
        
        return {
            "total_emails": len(request.emails),
            "results": results,
            "model_used": request.model_type.upper(),
            "total_latency_ms": round(latency, 2),
            "avg_latency_ms": round(latency / len(request.emails), 2)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models/info")
def get_models_info():
    """Get information about available models"""
    return {
        "models": [
            {
                "name": "Linear SVM",
                "type": "svm",
                "accuracy": 0.9892,
                "f1_score": 0.9892,
                "latency_ms": 0.75,
                "description": "Best overall - highest accuracy and fastest"
            },
            {
                "name": "BERT Transformer",
                "type": "bert",
                "accuracy": 0.9821,
                "f1_score": 0.9821,
                "latency_ms": 50.89,
                "description": "State-of-the-art deep learning model"
            }
        ],
        "recommendation": "Use SVM for production (best speed + accuracy)"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)