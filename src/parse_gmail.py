import mailbox
import email
import pandas as pd
from pathlib import Path
import re
from tqdm import tqdm

def parse_mbox_file(mbox_path):
    """Parse Gmail MBOX export"""
    
    print(f"Loading emails from {mbox_path}...")
    mbox = mailbox.mbox(mbox_path)
    
    emails = []
    
    for message in tqdm(mbox):
        try:
            # Extract fields
            subject = message.get('Subject', '')
            from_addr = message.get('From', '')
            to_addr = message.get('To', '')
            date = message.get('Date', '')
            
            # Extract body
            body = ""
            if message.is_multipart():
                for part in message.walk():
                    if part.get_content_type() == "text/plain":
                        try:
                            body = part.get_payload(decode=True).decode('utf-8', errors='ignore')
                        except:
                            pass
                        break
            else:
                try:
                    body = message.get_payload(decode=True).decode('utf-8', errors='ignore')
                except:
                    body = str(message.get_payload())
            
            # Check if it's in spam folder or has spam indicators
            # You'll need to manually label some or use heuristics
            is_spam = 0  # Default to ham
            
            # Heuristic: check for spam folder
            gmail_labels = message.get('X-Gmail-Labels', '')
            if 'Spam' in gmail_labels or 'SPAM' in gmail_labels:
                is_spam = 1
            
            emails.append({
                'subject': subject,
                'from': from_addr,
                'to': to_addr,
                'date': date,
                'body': body,
                'text': f"{subject} {body}",
                'is_spam': is_spam,
                'gmail_labels': gmail_labels
            })
            
        except Exception as e:
            continue
    
    return emails

def clean_text(text):
    """Clean email text"""
    if not text:
        return ""
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove special characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

def main():
    """Main pipeline"""
    
    # Path to your MBOX file
    mbox_path = "data/raw/gmail.mbox"  # Update this path
    
    if not Path(mbox_path).exists():
        print(f"Error: {mbox_path} not found!")
        print("Please download your Gmail data from Google Takeout")
        print("Place the .mbox file in data/raw/")
        return
    
    # Parse emails
    emails = parse_mbox_file(mbox_path)
    
    print(f"\nParsed {len(emails)} emails")
    
    # Create DataFrame
    df = pd.DataFrame(emails)
    
    # Clean text
    print("Cleaning text...")
    df['cleaned_text'] = df['text'].apply(clean_text)
    
    # Remove very short emails
    df = df[df['cleaned_text'].str.len() > 10]
    
    print(f"\nAfter filtering: {len(df)} emails")
    print(f"Spam: {df['is_spam'].sum()}")
    print(f"Ham: {len(df) - df['is_spam'].sum()}")
    
    # Save
    output_path = "data/processed/gmail_emails.csv"
    df.to_csv(output_path, index=False)
    print(f"\n✓ Saved to {output_path}")
    
    # Show sample
    print("\nSample emails:")
    print(df[['subject', 'from', 'is_spam']].head(10))

if __name__ == "__main__":
    main()