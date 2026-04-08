import requests
import json

BASE_URL = "http://localhost:8000"

def test_single_email():
    """Test single email classification"""
    
    print("\n" + "="*50)
    print("TEST 1: Single Email Classification (SVM)")
    print("="*50)
    
    email_text = "Congratulations! You've won a free iPhone! Click here to claim your prize now!"
    
    response = requests.post(
        f"{BASE_URL}/classify",
        json={"text": email_text, "model_type": "svm"}
    )
    
    result = response.json()
    print(f"Email: {email_text}")
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.4f}")
    print(f"Latency: {result['latency_ms']}ms")

def test_batch_emails():
    """Test batch classification"""
    
    print("\n" + "="*50)
    print("TEST 2: Batch Email Classification (SVM)")
    print("="*50)
    
    emails = [
        "Meeting at 3pm tomorrow in conference room B",
        "FREE VIAGRA!!! Click here NOW!!!",
        "Can you review my pull request?",
        "You have won $1,000,000! Claim now!"
    ]
    
    response = requests.post(
        f"{BASE_URL}/classify_batch",
        json={"emails": emails, "model_type": "svm"}
    )
    
    result = response.json()
    print(f"Total emails: {result['total_emails']}")
    print(f"Avg latency: {result['avg_latency_ms']}ms")
    print("\nResults:")
    for r in result['results']:
        print(f"  - {r['email'][:50]}... → {r['prediction']} ({r['confidence']:.2%})")

def test_models_info():
    """Test models info endpoint"""
    
    print("\n" + "="*50)
    print("TEST 3: Models Information")
    print("="*50)
    
    response = requests.get(f"{BASE_URL}/models/info")
    result = response.json()
    
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    print("\n🚀 Testing AI Email Assistant API...")
    
    try:
        # Test health check
        response = requests.get(BASE_URL)
        print("\n✓ API is running!")
        
        # Run tests
        test_single_email()
        test_batch_emails()
        test_models_info()
        
        print("\n✅ All tests passed!")
        
    except requests.exceptions.ConnectionError:
        print("\n❌ API is not running. Start it with: uvicorn api.main:app --reload")