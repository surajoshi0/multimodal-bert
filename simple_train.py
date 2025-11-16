"""
Simple BERT Text-Only Training Script
Trains BERT on medical text data (impressions/findings)
"""

import torch
import pandas as pd
import os
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from tqdm import tqdm, trange
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

# ============ CONFIGURATION ============
BATCH_SIZE = 8  # Smaller batch size for stability (CPU)
EPOCHS = 3
MAX_SEQ_LENGTH = 256
LEARNING_RATE = 2e-5
WARMUP_STEPS = 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GRADIENT_ACCUMULATION_STEPS = 2  # Simulate larger batch size

# Data paths
DATA_DIR = "./data/csv"
TRAIN_FILE = "image_labels_impression_train.csv"
VAL_FILE = "image_labels_impression_val.csv"
TEST_FILE = "image_labels_impression_test.csv"

OUTPUT_DIR = "./model_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Using device: {DEVICE}")
print(f"GPU Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ============ LOAD DATA ============
print("\n=== Loading Data ===")
train_df = pd.read_csv(os.path.join(DATA_DIR, TRAIN_FILE))
val_df = pd.read_csv(os.path.join(DATA_DIR, VAL_FILE))
test_df = pd.read_csv(os.path.join(DATA_DIR, TEST_FILE))

print(f"Training samples: {len(train_df)}")
print(f"Validation samples: {len(val_df)}")
print(f"Test samples: {len(test_df)}")
print(f"\nSample text: {train_df['text'].iloc[0]}")
print(f"Sample label: {train_df['label'].iloc[0]}")

# ============ TOKENIZE DATA ============
print("\n=== Tokenizing Data ===")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

def tokenize_texts(texts, labels, max_length=256):
    """Tokenize text data"""
    input_ids = []
    attention_masks = []
    
    for text in texts:
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

# Tokenize all data
train_inputs, train_masks, train_labels = tokenize_texts(
    train_df['text'].values, 
    train_df['label']
)
val_inputs, val_masks, val_labels = tokenize_texts(
    val_df['text'].values, 
    val_df['label']
)
test_inputs, test_masks, test_labels = tokenize_texts(
    test_df['text'].values, 
    test_df['label']
)

print("Tokenization complete!")
print(f"Train inputs shape: {train_inputs.shape}")

# ============ CREATE DATASETS & DATALOADERS ============
print("\n=== Creating DataLoaders ===")
train_dataset = TensorDataset(train_inputs, train_masks, train_labels)
val_dataset = TensorDataset(val_inputs, val_masks, val_labels)
test_dataset = TensorDataset(test_inputs, test_masks, test_labels)

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
test_loader = DataLoader(
    test_dataset,
    sampler=SequentialSampler(test_dataset),
    batch_size=BATCH_SIZE
)

# ============ LOAD MODEL ============
print("\n=== Loading BERT Model ===")
model = AutoModelForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=2  # Binary classification (0 or 1)
)
model.to(DEVICE)

# ============ SETUP OPTIMIZER & SCHEDULER ============
total_steps = len(train_loader) * EPOCHS
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, eps=1e-8)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=WARMUP_STEPS,
    num_training_steps=total_steps
)

# ============ TRAINING FUNCTION ============
def train_epoch(model, train_loader, optimizer, scheduler, device, accumulation_steps=1):
    """Train one epoch"""
    model.train()
    total_loss = 0
    optimizer.zero_grad()
    
    progress_bar = tqdm(train_loader, desc="Training")
    for step, batch in enumerate(progress_bar):
        input_ids, attention_mask, labels = [b.to(device) for b in batch]
        
        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        loss = outputs.loss
        
        # Scale loss by accumulation steps
        loss = loss / accumulation_steps
        
        # Backward pass
        loss.backward()
        
        # Update weights after accumulation steps
        if (step + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * accumulation_steps
        progress_bar.set_postfix({'loss': loss.item() * accumulation_steps})
    
    return total_loss / len(train_loader)

# ============ EVALUATION FUNCTION ============
def evaluate(model, val_loader, device):
    """Evaluate model"""
    model.eval()
    predictions = []
    true_labels = []
    total_loss = 0
    
    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc="Evaluating")
        for batch in progress_bar:
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
    f1 = f1_score(true_labels, predictions)
    
    return {
        'loss': total_loss / len(val_loader),
        'accuracy': accuracy,
        'f1': f1
    }

# ============ TRAINING LOOP ============
print("\n=== Starting Training ===")
print(f"Total steps: {total_steps}")
print(f"Epochs: {EPOCHS}\n")

best_val_accuracy = 0
for epoch in trange(EPOCHS, desc="Epoch"):
    print(f"\n--- Epoch {epoch + 1}/{EPOCHS} ---")
    
    # Train
    avg_loss = train_epoch(model, train_loader, optimizer, scheduler, DEVICE, GRADIENT_ACCUMULATION_STEPS)
    print(f"Average training loss: {avg_loss:.4f}")
    
    # Validate
    val_results = evaluate(model, val_loader, DEVICE)
    print(f"Validation Loss: {val_results['loss']:.4f}")
    print(f"Validation Accuracy: {val_results['accuracy']:.4f}")
    print(f"Validation F1: {val_results['f1']:.4f}")
    
    # Save best model
    if val_results['accuracy'] > best_val_accuracy:
        best_val_accuracy = val_results['accuracy']
        model.save_pretrained(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
        print(f"✓ Model saved with accuracy: {best_val_accuracy:.4f}")

# ============ TEST EVALUATION ============
print("\n=== Testing Model ===")
test_results = evaluate(model, test_loader, DEVICE)
print(f"Test Loss: {test_results['loss']:.4f}")
print(f"Test Accuracy: {test_results['accuracy']:.4f}")
print(f"Test F1: {test_results['f1']:.4f}")

print(f"\n✓ Training complete! Model saved to: {OUTPUT_DIR}")
