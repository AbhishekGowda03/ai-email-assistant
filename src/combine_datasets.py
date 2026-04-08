import pandas as pd
import numpy as np

def main():
    """Combine Gmail ham + SpamAssassin spam"""
    
    print("="*60)
    print("COMBINING DATASETS")
    print("="*60)
    
    # Load Gmail (all ham)
    print("\nLoading Gmail emails...")
    gmail_df = pd.read_csv('data/processed/gmail_emails.csv')
    gmail_ham = gmail_df[gmail_df['is_spam'] == 0].copy()
    
    print(f"✓ Gmail ham emails: {len(gmail_ham)}")
    
    # Load original SpamAssassin dataset
    print("\nLoading SpamAssassin dataset...")
    original_df = pd.read_csv('data/processed/emails_dataset.csv')
    spam_df = original_df[original_df['is_spam'] == 1].copy()
    
    print(f"✓ SpamAssassin spam emails: {len(spam_df)}")
    
    # Balance: take equal amounts
    min_count = min(len(gmail_ham), len(spam_df))
    
    print(f"\n{'='*60}")
    print(f"Balancing to {min_count} emails per class...")
    print(f"{'='*60}")
    
    # Sample equally from both
    ham_balanced = gmail_ham.sample(n=min_count, random_state=42)
    spam_balanced = spam_df.sample(n=min_count, random_state=42)
    
    # Ensure columns match
    print("\nAligning columns...")
    
    # Get common columns
    common_cols = ['text', 'cleaned_text', 'is_spam']
    
    # Add missing columns if needed
    if 'subject' in ham_balanced.columns and 'subject' in spam_balanced.columns:
        common_cols.insert(0, 'subject')
    
    if 'body' in ham_balanced.columns and 'body' in spam_balanced.columns:
        common_cols.insert(1, 'body')
    
    # Select only common columns
    ham_final = ham_balanced[common_cols].copy()
    spam_final = spam_balanced[common_cols].copy()
    
    # Combine
    print("\nCombining datasets...")
    combined = pd.concat([ham_final, spam_final], ignore_index=True)
    
    # Shuffle
    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Add synthetic labels for multi-task learning
    print("\nAdding multi-task labels...")
    combined['is_important'] = combined['is_spam'].apply(lambda x: 0 if x == 1 else 1)
    combined['priority_score'] = combined['is_spam'].apply(
        lambda x: 0.0 if x == 1 else np.random.uniform(0.5, 1.0)
    )
    
    print(f"\n{'='*60}")
    print("FINAL COMBINED DATASET")
    print(f"{'='*60}")
    print(f"Total emails: {len(combined)}")
    print(f"Spam: {combined['is_spam'].sum()} ({combined['is_spam'].mean()*100:.1f}%)")
    print(f"Ham: {(combined['is_spam']==0).sum()} ({(1-combined['is_spam'].mean())*100:.1f}%)")
    print(f"\nColumns: {list(combined.columns)}")
    
    # Save
    output_path = 'data/processed/combined_emails.csv'
    combined.to_csv(output_path, index=False)
    print(f"\n✓ Saved to {output_path}")
    
    # Show samples
    print(f"\n{'='*60}")
    print("SAMPLE EMAILS FROM COMBINED DATASET")
    print(f"{'='*60}")
    
    print("\nSample HAM (your Gmail):")
    ham_samples = combined[combined['is_spam']==0].head(3)
    for idx, row in ham_samples.iterrows():
        text_preview = row['text'][:100] + "..." if len(row['text']) > 100 else row['text']
        print(f"\n  Text: {text_preview}")
        print(f"  Label: HAM")
    
    print("\nSample SPAM (SpamAssassin):")
    spam_samples = combined[combined['is_spam']==1].head(3)
    for idx, row in spam_samples.iterrows():
        text_preview = row['text'][:100] + "..." if len(row['text']) > 100 else row['text']
        print(f"\n  Text: {text_preview}")
        print(f"  Label: SPAM")
    
    print(f"\n{'='*60}")
    print("✅ DATASET COMBINATION COMPLETE!")
    print(f"{'='*60}")
    print("\nNext steps:")
    print("1. Run: python src/train_test_split_combined.py")
    print("2. Re-train all models on YOUR personalized dataset!")

if __name__ == "__main__":
    main()