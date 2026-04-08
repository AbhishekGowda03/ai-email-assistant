import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import time

def load_data():
    """Load train and test data"""
    train_df = pd.read_csv('data/processed/train.csv')
    test_df = pd.read_csv('data/processed/test.csv')
    
    return train_df, test_df

def create_features(X_train, X_test, max_features=5000):
    """Create TF-IDF features"""
    print(f"Creating TF-IDF features (max_features={max_features})...")
    
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        min_df=2,
        max_df=0.8,
        ngram_range=(1, 2)  # unigrams and bigrams
    )
    
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    print(f"Feature matrix shape: {X_train_tfidf.shape}")
    
    # Save vectorizer
    joblib.dump(vectorizer, 'models/baseline/tfidf_vectorizer.pkl')
    
    return X_train_tfidf, X_test_tfidf, vectorizer

def train_and_evaluate(model, model_name, X_train, X_test, y_train, y_test):
    """Train a model and evaluate it"""
    print(f"\n{'='*50}")
    print(f"Training {model_name}...")
    print(f"{'='*50}")
    
    # Train
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    # Predict
    start_time = time.time()
    y_pred = model.predict(X_test)
    inference_time = time.time() - start_time
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"\nResults for {model_name}:")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"Train time: {train_time:.2f}s")
    print(f"Inference time: {inference_time:.4f}s")
    print(f"Latency per email: {(inference_time/len(y_test))*1000:.2f}ms")
    
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Ham', 'Spam'], 
                yticklabels=['Ham', 'Spam'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'results/plots/cm_{model_name.lower().replace(" ", "_")}.png')
    plt.close()
    
    # Save model
    model_path = f'models/baseline/{model_name.lower().replace(" ", "_")}.pkl'
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
    
    return {
        'model_name': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'train_time': train_time,
        'inference_time': inference_time,
        'latency_ms': (inference_time/len(y_test))*1000
    }

def main():
    """Main training pipeline"""
    
    # Load data
    print("Loading data...")
    train_df, test_df = load_data()
    
    X_train = train_df['text']
    X_test = test_df['text']
    y_train = train_df['is_spam']
    y_test = test_df['is_spam']
    
    # Create TF-IDF features
    X_train_tfidf, X_test_tfidf, vectorizer = create_features(X_train, X_test)
    
    # Define models
    models = {
        'Naive Bayes': MultinomialNB(),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Linear SVM': LinearSVC(max_iter=1000, random_state=42)
    }
    
    # Train and evaluate all models
    results = []
    for model_name, model in models.items():
        result = train_and_evaluate(
            model, model_name, 
            X_train_tfidf, X_test_tfidf, 
            y_train, y_test
        )
        results.append(result)
    
    # Compare models
    results_df = pd.DataFrame(results)
    results_df.to_csv('results/metrics/baseline_results.csv', index=False)
    
    print("\n" + "="*50)
    print("BASELINE MODELS COMPARISON")
    print("="*50)
    print(results_df.to_string(index=False))
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Accuracy
    axes[0, 0].bar(results_df['model_name'], results_df['accuracy'])
    axes[0, 0].set_title('Accuracy Comparison')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_ylim([0.8, 1.0])
    
    # F1 Score
    axes[0, 1].bar(results_df['model_name'], results_df['f1'])
    axes[0, 1].set_title('F1 Score Comparison')
    axes[0, 1].set_ylabel('F1 Score')
    axes[0, 1].set_ylim([0.8, 1.0])
    
    # Training Time
    axes[1, 0].bar(results_df['model_name'], results_df['train_time'])
    axes[1, 0].set_title('Training Time')
    axes[1, 0].set_ylabel('Time (seconds)')
    
    # Latency
    axes[1, 1].bar(results_df['model_name'], results_df['latency_ms'])
    axes[1, 1].set_title('Inference Latency (per email)')
    axes[1, 1].set_ylabel('Latency (ms)')
    
    plt.tight_layout()
    plt.savefig('results/plots/baseline_comparison.png', dpi=100)
    plt.show()
    
    print(f"\nAll results saved to results/metrics/ and results/plots/")

if __name__ == "__main__":
    main()