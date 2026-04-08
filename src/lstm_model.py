import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import time

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class EmailDataset(Dataset):
    """Custom Dataset for emails"""
    
    def __init__(self, texts, labels, vocab, max_len=200):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts.iloc[idx]
        label = self.labels.iloc[idx]
        
        # Tokenize and convert to indices
        tokens = text.split()[:self.max_len]
        indices = [self.vocab.get(token, self.vocab['<UNK>']) for token in tokens]
        
        # Pad sequence
        if len(indices) < self.max_len:
            indices += [self.vocab['<PAD>']] * (self.max_len - len(indices))
        
        return torch.tensor(indices, dtype=torch.long), torch.tensor(label, dtype=torch.float)

class LSTMClassifier(nn.Module):
    """LSTM-based email classifier"""
    
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, num_layers=2, dropout=0.3):
        super(LSTMClassifier, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, 1)  # *2 for bidirectional
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # x: (batch_size, seq_len)
        embedded = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # Use last hidden state from both directions
        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)  # (batch_size, hidden_dim*2)
        
        dropped = self.dropout(hidden)
        output = self.fc(dropped)
        return self.sigmoid(output).squeeze()

def build_vocab(texts, max_vocab_size=10000):
    """Build vocabulary from texts"""
    from collections import Counter
    
    word_counts = Counter()
    for text in texts:
        word_counts.update(text.split())
    
    # Most common words
    most_common = word_counts.most_common(max_vocab_size - 2)
    
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for idx, (word, _) in enumerate(most_common, start=2):
        vocab[word] = idx
    
    return vocab

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

def evaluate(model, dataloader, criterion, device):
    """Evaluate model"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            preds = (outputs > 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    
    return avg_loss, accuracy, all_preds, all_labels

def main():
    """Main training pipeline"""
    
    # Load data
    print("Loading data...")
    train_df = pd.read_csv('data/processed/train.csv')
    test_df = pd.read_csv('data/processed/test.csv')
    
    # Build vocabulary
    print("Building vocabulary...")
    vocab = build_vocab(train_df['text'], max_vocab_size=10000)
    print(f"Vocabulary size: {len(vocab)}")
    
    # Create datasets
    max_len = 200
    train_dataset = EmailDataset(train_df['text'], train_df['is_spam'], vocab, max_len)
    test_dataset = EmailDataset(test_df['text'], test_df['is_spam'], vocab, max_len)
    
    # Create dataloaders
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    print("\nInitializing LSTM model...")
    model = LSTMClassifier(
        vocab_size=len(vocab),
        embedding_dim=128,
        hidden_dim=256,
        num_layers=2,
        dropout=0.3
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training
    num_epochs = 10
    print(f"\nTraining for {num_epochs} epochs...")
    
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    best_val_acc = 0
    start_time = time.time()
    
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = evaluate(model, test_loader, criterion, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Val Accuracy: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'models/lstm/best_lstm_model.pth')
            print(f"  ✓ Best model saved!")
        print()
    
    train_time = time.time() - start_time
    print(f"Total training time: {train_time:.2f}s")
    
    # Load best model for final evaluation
    model.load_state_dict(torch.load('models/lstm/best_lstm_model.pth'))
    
    # Final evaluation
    print("\n" + "="*50)
    print("FINAL EVALUATION")
    print("="*50)
    
    start_time = time.time()
    _, _, y_pred, y_true = evaluate(model, test_loader, criterion, device)
    inference_time = time.time() - start_time
    
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    print(f"\nResults for LSTM:")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"Train time: {train_time:.2f}s")
    print(f"Inference time: {inference_time:.4f}s")
    print(f"Latency per email: {(inference_time/len(y_true))*1000:.2f}ms")
    
    print(f"\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['Ham', 'Spam']))
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Ham', 'Spam'], 
                yticklabels=['Ham', 'Spam'])
    plt.title('Confusion Matrix - LSTM')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('results/plots/cm_lstm.png')
    plt.close()
    
    # Training curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(train_losses, label='Train Loss')
    axes[0].plot(val_losses, label='Val Loss')
    axes[0].set_title('Loss Curves')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    axes[1].plot(val_accuracies, label='Val Accuracy', marker='o')
    axes[1].set_title('Validation Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('results/plots/lstm_training_curves.png')
    plt.show()
    
    # Save results
    results = {
        'model_name': 'LSTM',
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'train_time': train_time,
        'inference_time': inference_time,
        'latency_ms': (inference_time/len(y_true))*1000
    }
    
    results_df = pd.DataFrame([results])
    results_df.to_csv('results/metrics/lstm_results.csv', index=False)
    
    print("\nResults saved!")

if __name__ == "__main__":
    main()