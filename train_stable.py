"""
Ultra-Simple BERT Training - CPU Stable Version
Uses older PyTorch APIs for compatibility
"""

import torch
import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')

# Set environment variables for stability
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from tqdm import tqdm
from sklearn.metrics import accuracy_score

# ============ CONFIG ============
BATCH_SIZE = 2  # Very small for CPU
EPOCHS = 1
MAX_LENGTH = 128
LR = 2e-5
DEVICE = "cpu"

DATA_DIR = "./data/csv"
OUTPUT_DIR = "./model_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("\n" + "="*60)
print("BERT TEXT CLASSIFICATION - STABLE VERSION")
print("="*60)

# ============ LOAD DATA ============
print("\n📁 Loading data...")
train_df = pd.read_csv(os.path.join(DATA_DIR, "image_labels_impression_train.csv"))
train_df = train_df.sample(200, random_state=42)  # Only 200 samples for testing
print(f"✓ Loaded {len(train_df)} samples")

# ============ TOKENIZE ============
print("\n🔤 Tokenizing...")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

input_ids = []
attention_masks = []

for text in tqdm(train_df['text'].values, desc="Tokenizing"):
    encoded = tokenizer.encode_plus(
        str(text)[:512],  # Limit text length
        add_special_tokens=True,
        max_length=MAX_LENGTH,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    input_ids.append(encoded['input_ids'].squeeze())
    attention_masks.append(encoded['attention_mask'].squeeze())

input_ids = torch.stack(input_ids)
attention_masks = torch.stack(attention_masks)
labels = torch.tensor(train_df['label'].values)

print(f"✓ Tokenized: {input_ids.shape}")

# ============ CREATE DATALOADER ============
print("\n📦 Creating dataloader...")
dataset = TensorDataset(input_ids, attention_masks, labels)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
print(f"✓ Created {len(dataloader)} batches")

# ============ LOAD MODEL ============
print("\n🤖 Loading BERT...")
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=2,
    output_hidden_states=False
)
model.to(DEVICE)
print("✓ Model loaded")

# ============ SETUP OPTIMIZER ============
print("\n⚙️  Setting up training...")
optimizer = AdamW(model.parameters(), lr=LR)
print("✓ Optimizer ready")

# ============ TRAINING LOOP ============
print("\n" + "="*60)
print("🚀 STARTING TRAINING")
print("="*60 + "\n")

for epoch in range(EPOCHS):
    print(f"Epoch {epoch + 1}/{EPOCHS}")
    
    model.train()
    total_loss = 0
    predictions = []
    true_labels = []
    
    for batch_idx, (batch_input_ids, batch_attention_masks, batch_labels) in enumerate(dataloader):
        batch_input_ids = batch_input_ids.to(DEVICE)
        batch_attention_masks = batch_attention_masks.to(DEVICE)
        batch_labels = batch_labels.to(DEVICE)
        
        # Forward pass
        outputs = model(
            input_ids=batch_input_ids,
            attention_mask=batch_attention_masks,
            labels=batch_labels
        )
        
        loss = outputs.loss
        logits = outputs.logits
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Get predictions
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        predictions.extend(preds)
        true_labels.extend(batch_labels.cpu().numpy())
        
        if (batch_idx + 1) % 20 == 0:
            print(f"  Batch {batch_idx + 1} - Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(true_labels, predictions)
    
    print(f"\nEpoch Results:")
    print(f"  Average Loss: {avg_loss:.4f}")
    print(f"  Accuracy: {accuracy:.4f}\n")

# ============ SAVE MODEL ============
print("💾 Saving model...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"✓ Model saved to {OUTPUT_DIR}")

print("\n" + "="*60)
print("✅ TRAINING COMPLETE!")
print("="*60)
print("\nTo run full training with all data, use: simple_train.py")
print("="*60 + "\n")
