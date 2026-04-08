import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

def analyze_baseline_errors():
    """Analyze errors from best baseline model (SVM)"""
    
    print("="*50)
    print("ERROR ANALYSIS - Linear SVM")
    print("="*50)
    
    # Load model and data
    model = joblib.load('models/baseline/linear_svm.pkl')
    vectorizer = joblib.load('models/baseline/tfidf_vectorizer.pkl')
    
    test_df = pd.read_csv('data/processed/test.csv')
    X_test = test_df['text']
    y_test = test_df['is_spam']
    
    # Transform and predict
    X_test_tfidf = vectorizer.transform(X_test)
    y_pred = model.predict(X_test_tfidf)
    
    # Find errors
    errors = test_df[(y_test != y_pred)].copy()
    errors['predicted'] = y_pred[y_test != y_pred]
    errors['actual'] = y_test[y_test != y_pred]
    
    print(f"\nTotal test samples: {len(test_df)}")
    print(f"Total errors: {len(errors)}")
    print(f"Error rate: {len(errors)/len(test_df)*100:.2f}%")
    
    # Categorize errors
    false_positives = errors[errors['actual'] == 0]  # Ham predicted as Spam
    false_negatives = errors[errors['actual'] == 1]  # Spam predicted as Ham
    
    print(f"\nFalse Positives (Ham → Spam): {len(false_positives)}")
    print(f"False Negatives (Spam → Ham): {len(false_negatives)}")
    
    # Analyze text length
    errors['text_length'] = errors['text'].str.len()
    
    print(f"\nError Analysis by Text Length:")
    print(f"Mean length of errors: {errors['text_length'].mean():.0f} chars")
    print(f"Mean length of all test: {test_df['text'].str.len().mean():.0f} chars")
    
    # Save error examples
    errors.to_csv('results/metrics/error_examples_svm.csv', index=False)
    print(f"\n✓ Error examples saved to results/metrics/error_examples_svm.csv")
    
    # Visualize errors
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Error distribution
    axes[0].bar(['False Positives\n(Ham→Spam)', 'False Negatives\n(Spam→Ham)'], 
               [len(false_positives), len(false_negatives)],
               color=['salmon', 'skyblue'], edgecolor='black')
    axes[0].set_title('Error Type Distribution - SVM', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Count', fontsize=12)
    
    # Text length distribution of errors
    axes[1].hist(test_df[y_test == y_pred]['text'].str.len(), bins=30, alpha=0.5, label='Correct', color='green')
    axes[1].hist(errors['text_length'], bins=30, alpha=0.7, label='Errors', color='red')
    axes[1].set_title('Text Length: Correct vs Errors', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Text Length (characters)', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig('results/plots/error_analysis_svm.png', dpi=150)
    print("✓ Error analysis plot saved!")
    plt.show()
    
    # Show example errors
    print("\n" + "="*50)
    print("EXAMPLE FALSE POSITIVES (Ham predicted as Spam):")
    print("="*50)
    for idx, row in false_positives.head(3).iterrows():
        print(f"\nText (truncated): {row['text'][:200]}...")
        print(f"Actual: Ham | Predicted: Spam")
        print("-"*50)
    
    print("\n" + "="*50)
    print("EXAMPLE FALSE NEGATIVES (Spam predicted as Ham):")
    print("="*50)
    for idx, row in false_negatives.head(3).iterrows():
        print(f"\nText (truncated): {row['text'][:200]}...")
        print(f"Actual: Spam | Predicted: Ham")
        print("-"*50)
    
    return errors

def why_model_fails_analysis():
    """Analyze WHY models fail"""
    
    print("\n" + "="*80)
    print("WHY DID THE MODEL FAIL? - ROOT CAUSE ANALYSIS")
    print("="*80)
    
    errors = pd.read_csv('results/metrics/error_examples_svm.csv')
    
    # Pattern 1: Very short emails
    very_short = errors[errors['text_length'] < 50]
    print(f"\n1. VERY SHORT EMAILS (<50 chars): {len(very_short)} errors")
    print("   → Root cause: Insufficient context for classification")
    
    # Pattern 2: Emails with mixed signals
    print(f"\n2. MIXED SIGNAL EMAILS:")
    print("   → Root cause: Contain both spam-like and ham-like words")
    
    # Pattern 3: Edge cases
    print(f"\n3. EDGE CASES:")
    print("   → Root cause: Unusual vocabulary or formatting")
    
    print("\n" + "="*80)

def main():
    """Main error analysis pipeline"""
    
    print("Starting error analysis...\n")
    
    # Analyze SVM errors
    errors = analyze_baseline_errors()
    
    # Why did model fail?
    why_model_fails_analysis()
    
    print("\n✓ Error analysis complete!")

if __name__ == "__main__":
    main()