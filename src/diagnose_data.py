import pandas as pd

# Load original dataset
df = pd.read_csv('data/processed/emails_dataset.csv')

print("="*50)
print("DATA DIAGNOSIS")
print("="*50)

print(f"\nTotal emails: {len(df)}")
print(f"Spam count: {df['is_spam'].sum()}")
print(f"Ham count: {(df['is_spam'] == 0).sum()}")
print(f"Spam ratio: {df['is_spam'].mean():.2%}")

print("\n--- Missing/Empty Text Analysis ---")
print(f"Missing cleaned_text: {df['cleaned_text'].isna().sum()}")
print(f"Empty cleaned_text (len=0): {(df['cleaned_text'].str.len() == 0).sum()}")
print(f"Very short (<5 chars): {(df['cleaned_text'].str.len() < 5).sum()}")
print(f"Short (<10 chars): {(df['cleaned_text'].str.len() < 10).sum()}")

print("\n--- By Class ---")
spam_df = df[df['is_spam'] == 1]
ham_df = df[df['is_spam'] == 0]

print(f"\nSPAM emails:")
print(f"  Total: {len(spam_df)}")
print(f"  Missing cleaned_text: {spam_df['cleaned_text'].isna().sum()}")
print(f"  Empty cleaned_text: {(spam_df['cleaned_text'].str.len() == 0).sum()}")
print(f"  Mean text length: {spam_df['cleaned_text'].str.len().mean():.0f}")

print(f"\nHAM emails:")
print(f"  Total: {len(ham_df)}")
print(f"  Missing cleaned_text: {ham_df['cleaned_text'].isna().sum()}")
print(f"  Empty cleaned_text: {(ham_df['cleaned_text'].str.len() == 0).sum()}")
print(f"  Mean text length: {ham_df['cleaned_text'].str.len().mean():.0f}")

# Check what gets filtered out
filtered_df = df.dropna(subset=['cleaned_text'])
filtered_df = filtered_df[filtered_df['cleaned_text'].str.len() > 5]

print(f"\n--- After Filtering ---")
print(f"Remaining emails: {len(filtered_df)}")
print(f"Spam ratio: {filtered_df['is_spam'].mean():.2%}")
print(f"Lost {len(df) - len(filtered_df)} emails total")
print(f"  Lost spam: {spam_df['is_spam'].sum() - filtered_df['is_spam'].sum()}")
print(f"  Lost ham: {len(ham_df) - (len(filtered_df) - filtered_df['is_spam'].sum())}")