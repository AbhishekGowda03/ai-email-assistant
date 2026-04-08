import os
import email
import pandas as pd
import re
from pathlib import Path
import nltk
from tqdm import tqdm

# Download NLTK data
nltk.download('stopwords')
nltk.download('punkt')

def parse_email_file(filepath):
    """Parse a single email file"""
    try:
        with open(filepath, 'r', encoding='latin-1') as f:
            msg = email.message_from_file(f)
            
            # Extract subject
            subject = msg.get('Subject', '')
            
            # Extract body
            body = ""
            if msg.is_multipart():
                for part in msg.walk():
                    if part.get_content_type() == "text/plain":
                        body = part.get_payload(decode=True)
                        if body:
                            body = body.decode('latin-1', errors='ignore')
                        break
            else:
                body = msg.get_payload(decode=True)
                if body:
                    body = body.decode('latin-1', errors='ignore')
            
            return {
                'subject': subject,
                'body': body,
                'text': f"{subject} {body}"
            }
    except Exception as e:
        return None

def load_emails_from_folder(folder_path, label):
    """Load all emails from a folder"""
    emails = []
    folder = Path(folder_path)
    
    files = list(folder.glob('*'))
    print(f"Loading {len(files)} emails from {folder_path}...")
    
    for filepath in tqdm(files):
        if filepath.is_file():
            parsed = parse_email_file(filepath)
            if parsed:
                parsed['label'] = label
                parsed['is_spam'] = 1 if label == 'spam' else 0
                emails.append(parsed)
    
    return emails

def clean_text(text):
    """Clean email text"""
    if not text:
        return ""
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

def create_dataset():
    """Create the complete dataset"""
    
    # Find the extracted folders
    spam_folder = "data/raw/spam_2"
    ham_folder = "data/raw/easy_ham"
    
    # Load spam emails
    spam_emails = load_emails_from_folder(spam_folder, 'spam')
    
    # Load ham emails
    ham_emails = load_emails_from_folder(ham_folder, 'ham')
    
    # Combine
    all_emails = spam_emails + ham_emails
    
    # Create DataFrame
    df = pd.DataFrame(all_emails)
    
    # Clean text
    print("Cleaning text...")
    df['cleaned_text'] = df['text'].apply(clean_text)
    
    # Add synthetic labels for multi-task learning
    # Important: ham emails are important (for now, we'll improve this later)
    df['is_important'] = df['is_spam'].apply(lambda x: 0 if x == 1 else 1)
    
    # Priority score (0-1, spam = 0, ham = random for now)
    import numpy as np
    df['priority_score'] = df['is_spam'].apply(
        lambda x: 0.0 if x == 1 else np.random.uniform(0.5, 1.0)
    )
    
    # Save to CSV
    output_path = "data/processed/emails_dataset.csv"
    df.to_csv(output_path, index=False)
    print(f"\nDataset saved to {output_path}")
    print(f"Total emails: {len(df)}")
    print(f"Spam: {df['is_spam'].sum()}")
    print(f"Ham: {len(df) - df['is_spam'].sum()}")
    
    return df

if __name__ == "__main__":
    df = create_dataset()
    print("\nFirst few rows:")
    print(df.head())
    print("\nDataset info:")
    print(df.info())