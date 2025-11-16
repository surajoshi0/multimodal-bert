"""
Fast BERT Text Training Script - CPU Optimized
Trains quickly on smaller subset for demonstration
Good baseline before full training
"""

import torch
import pandas as pd
import os
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from tqdm import tqdm, trange
from sklearn.metrics import accuracy_score, f1_score
import warnings
warnings.filterwarnings('ignore')

# ============ CONFIGURATION ============
# Smaller sizes for fast CPU training
BATCH_SIZE = 4
EPOCHS = 1  # Just 1 epoch for quick test
MAX_SEQ_LENGTH = 128  # Shorter sequences for speed
LEARNING_RATE = 2e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAMPLE_SIZE = 500  # Use only 500 samples for quick test

# Data paths
DATA_DIR = "./data/csv"
TRAIN_FILE = "image_labels_impression_train.csv"
VAL_FILE = "image_labels_impression_val.csv"
TEST_FILE = "image_labels_impression_test.csv"

OUTPUT_DIR = "./model_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"\n{'='*60}")
print(f"BERT TEXT-ONLY TRAINING (FAST MODE)")
print(f"{'='*60}")
print(f"Device: {DEVICE}")
print(f"GPU Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Batch Size: {BATCH_SIZE}")
print(f"Epochs: {EPOCHS}")
print(f"Sample Size: {SAMPLE_SIZE}")
print(f"{'='*60}\n")

# ============ LOAD DATA ============
print("📁 Loading Data...")
train_df = pd.read_csv(os.path.join(DATA_DIR, TRAIN_FILE))
val_df = pd.read_csv(os.path.join(DATA_DIR, VAL_FILE))

# Use smaller sample for fast training
train_df = train_df.sample(min(SAMPLE_SIZE, len(train_df)), random_state=42)
val_df = val_df.sample(min(SAMPLE_SIZE//3, len(val_df)), random_state=42)

print(f"  Training samples: {len(train_df)}")
print(f"  Validation samples: {len(val_df)}")
print(f"  Sample text: {train_df['text'].iloc[0][:80]}...")
print(f"  Sample label: {train_df['label'].iloc[0]}\n")

# ============ TOKENIZE DATA ============
print("🔤 Tokenizing texts...")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

def tokenize_texts(texts, labels, max_length=128):
    """Tokenize text data"""
    input_ids = []
    attention_masks = []
    
    for i, text in enumerate(texts):
        if (i + 1) % 100 == 0:
            print(f"  Tokenized {i + 1}/{len(texts)}")
        
        encoded = tokenizer.encode_plus(
            str(text),
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )
        input_ids.append(encoded['input_ids'])
        attention_masks.append(encoded['attention_mask'])
    
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels_tensor = torch.tensor(labels.values, dtype=torch.long)
    
    return input_ids, attention_masks, labels_tensor

train_inputs, train_masks, train_labels = tokenize_texts(
    train_df['text'].values, 
    train_df['label']
)
val_inputs, val_masks, val_labels = tokenize_texts(
    val_df['text'].values, 
    val_df['label']
)

print(f"✓ Tokenization complete!")
print(f"  Train inputs shape: {train_inputs.shape}\n")

# ============ CREATE DATASETS & DATALOADERS ============
print("📦 Creating DataLoaders...")
train_dataset = TensorDataset(train_inputs, train_masks, train_labels)
val_dataset = TensorDataset(val_inputs, val_masks, val_labels)

train_loader = DataLoader(
    train_dataset,
    sampler=RandomSampler(train_dataset),
    batch_size=BATCH_SIZE
)
val_loader = DataLoader(
    val_dataset,
    sampler=SequentialSampler(val_dataset),
    batch_size=BATCH_SIZE
)

print(f"✓ DataLoaders created!\n")

# ============ LOAD MODEL ============
print("🤖 Loading BERT Model...")
model = AutoModelForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=2
)
model.to(DEVICE)
print(f"✓ Model loaded!\n")

# ============ SETUP OPTIMIZER ============
print("⚙️  Setting up optimizer...")
total_steps = len(train_loader) * EPOCHS
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, eps=1e-8)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)
print(f"✓ Optimizer ready!\n")

# ============ TRAINING FUNCTION ============
def train_epoch(model, train_loader, optimizer, scheduler, device):
    """Train one epoch"""
    model.train()
    total_loss = 0
    
    for batch_idx, batch in enumerate(train_loader):
        input_ids, attention_mask, labels = [b.to(device) for b in batch]
        
        optimizer.zero_grad()
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        loss = outputs.loss
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        
        if (batch_idx + 1) % max(1, len(train_loader)//5) == 0:
            print(f"  Batch {batch_idx + 1}/{len(train_loader)} - Loss: {loss.item():.4f}")
    
    return total_loss / len(train_loader)

# ============ EVALUATION FUNCTION ============
def evaluate(model, val_loader, device):
    """Evaluate model"""
    model.eval()
    predictions = []
    true_labels = []
    total_loss = 0
    
    with torch.no_grad():
        for batch in val_loader:
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss
            total_loss += loss.item()
            
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            predictions.extend(preds)
            true_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions, zero_division=0)
    
    return {
        'loss': total_loss / len(val_loader),
        'accuracy': accuracy,
        'f1': f1
    }

# ============ TRAINING LOOP ============
print(f"{'='*60}")
print("🚀 STARTING TRAINING")
print(f"{'='*60}\n")

best_val_accuracy = 0

for epoch in range(EPOCHS):
    print(f"--- Epoch {epoch + 1}/{EPOCHS} ---")
    
    # Train
    avg_loss = train_epoch(model, train_loader, optimizer, scheduler, DEVICE)
    print(f"Average training loss: {avg_loss:.4f}")
    
    # Validate
    val_results = evaluate(model, val_loader, DEVICE)
    print(f"Validation Loss: {val_results['loss']:.4f}")
    print(f"Validation Accuracy: {val_results['accuracy']:.4f}")
    print(f"Validation F1: {val_results['f1']:.4f}\n")
    
    # Save best model
    if val_results['accuracy'] > best_val_accuracy:
        best_val_accuracy = val_results['accuracy']
        model.save_pretrained(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
        print(f"✓ Model saved! Accuracy: {best_val_accuracy:.4f}\n")

print(f"{'='*60}")
print("✅ TRAINING COMPLETE!")
print(f"{'='*60}")
print(f"Model saved to: {OUTPUT_DIR}")
print(f"Best validation accuracy: {best_val_accuracy:.4f}")
print(f"\n🎯 NEXT STEPS:")
print("1. Run full training with all data")
print("2. Phase 2: Add image data and use MMBT")
print(f"{'='*60}\n")
