import pandas as pd
from sklearn.model_selection import train_test_split

def create_splits():
    """Create train/test splits from combined dataset"""
    
    print("="*60)
    print("CREATING TRAIN/TEST SPLITS")
    print("="*60)
    
    # Load combined data
    print("\nLoading combined dataset...")
    df = pd.read_csv('data/processed/combined_emails.csv')
    
    # Remove rows with missing cleaned_text
    df = df.dropna(subset=['cleaned_text'])
    df = df[df['cleaned_text'].str.len() > 5]
    
    print(f"Dataset size after filtering: {len(df)}")
    print(f"Spam: {df['is_spam'].sum()}")
    print(f"Ham: {len(df) - df['is_spam'].sum()}")
    
    # Split features and labels
    X = df['text']
    y_spam = df['is_spam']
    y_important = df['is_important']
    y_priority = df['priority_score']
    
    # 80-20 split
    print("\nCreating 80/20 train/test split...")
    X_train, X_test, y_spam_train, y_spam_test = train_test_split(
        X, y_spam, test_size=0.2, random_state=42, stratify=y_spam
    )
    
    _, _, y_important_train, y_important_test = train_test_split(
        X, y_important, test_size=0.2, random_state=42, stratify=y_spam
    )
    
    _, _, y_priority_train, y_priority_test = train_test_split(
        X, y_priority, test_size=0.2, random_state=42, stratify=y_spam
    )
    
    print(f"\nTrain size: {len(X_train)}")
    print(f"Test size: {len(X_test)}")
    print(f"Train spam ratio: {y_spam_train.mean():.2%}")
    print(f"Test spam ratio: {y_spam_test.mean():.2%}")
    
    # Save splits
    train_df = pd.DataFrame({
        'text': X_train,
        'is_spam': y_spam_train,
        'is_important': y_important_train,
        'priority_score': y_priority_train
    })
    
    test_df = pd.DataFrame({
        'text': X_test,
        'is_spam': y_spam_test,
        'is_important': y_important_test,
        'priority_score': y_priority_test
    })
    
    # IMPORTANT: Overwrite the old train/test files
    train_df.to_csv('data/processed/train.csv', index=False)
    test_df.to_csv('data/processed/test.csv', index=False)
    
    print("\n" + "="*60)
    print("✓ Train and test sets saved!")
    print("="*60)
    print(f"\nSaved to:")
    print(f"  - data/processed/train.csv ({len(train_df)} emails)")
    print(f"  - data/processed/test.csv ({len(test_df)} emails)")
    
    return train_df, test_df

if __name__ == "__main__":
    train_df, test_df = create_splits()