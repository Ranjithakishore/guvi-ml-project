# run this file in google colab
# !pip install fastapi uvicorn transformers torch nest-asyncio pyngrok
# from google.colab import drive
# drive.mount('/content/drive')
# from google.colab import drive
# drive.flush_and_unmount()

import pandas as pd
import numpy as np
import re
import os
import time
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import hamming_loss, accuracy_score, f1_score, roc_auc_score

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

class ReviewDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts.tolist() if hasattr(texts, 'tolist') else list(texts)
        self.labels = labels.tolist() if hasattr(labels, 'tolist') else list(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.FloatTensor(self.labels[idx])
        }

def preprocess_text(text):
    """Clean and normalize text data."""
    # Convert to lowercase
    text = text.lower()

    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Remove extra whitespace
    text = ' '.join(text.split())

    return text

def create_labels(df):
    """Create multi-label categories based on review content with 10 categories."""
    labels = []

    for _, row in df.iterrows():
        review = row['combined_text'].lower()
        # Initialize 10 categories
        label = [0] * 10

        # 1. Product Quality
        if any(word in review for word in [
            'quality', 'durable', 'reliable', 'broken', 'defect', 'build', 'material',
            'sturdy', 'flimsy', 'solid', 'construction', 'craftsmanship', 'made well',
            'poor quality', 'high quality', 'wear and tear', 'lasting'
        ]):
            label[0] = 1

        # 2. Customer Service
        if any(word in review for word in [
            'service', 'support', 'help', 'customer', 'return', 'refund', 'warranty',
            'representative', 'agent', 'response', 'contact', 'assistance', 'helpful',
            'unhelpful', 'responsive', 'communication', 'customer care', 'service desk'
        ]):
            label[1] = 1

        # 3. Price
        if any(word in review for word in [
            'price', 'cost', 'expensive', 'cheap', 'value', 'worth', 'affordable',
            'overpriced', 'bargain', 'discount', 'deal', 'money', 'pricing',
            'investment', 'budget', 'premium', 'economical', 'pricey'
        ]):
            label[2] = 1

        # 4. Functionality
        if any(word in review for word in [
            'work', 'function', 'feature', 'performance', 'capability', 'operates',
            'working', 'functional', 'operation', 'performing', 'works well',
            'doesn\'t work', 'stopped working', 'malfunctioning', 'operational'
        ]):
            label[3] = 1

        # 5. Technical Issues
        if any(word in review for word in [
            'bug', 'error', 'crash', 'glitch', 'problem', 'issue', 'malfunction',
            'freeze', 'stuck', 'technical', 'software', 'hardware', 'failure',
            'not working', 'broken', 'repair', 'fix', 'troubleshoot'
        ]):
            label[4] = 1

        # 6. Shipping/Delivery
        if any(word in review for word in [
            'shipping', 'delivery', 'arrived', 'package', 'shipment', 'late',
            'damaged', 'tracking', 'carrier', 'box', 'packaging', 'shipped',
            'transit', 'arrival', 'delayed', 'on time', 'shipping speed'
        ]):
            label[5] = 1

        # 7. User Experience
        if any(word in review for word in [
            'easy', 'difficult', 'simple', 'complicated', 'intuitive', 'user friendly',
            'confusing', 'straightforward', 'complex', 'learning curve', 'usability',
            'convenient', 'inconvenient', 'experience', 'interface', 'accessibility'
        ]):
            label[6] = 1

        # 8. Product Compatibility
        if any(word in review for word in [
            'compatible', 'compatibility', 'works with', 'fit', 'fits', 'connection',
            'connect', 'paired', 'sync', 'integration', 'supported', 'incompatible',
            'version', 'system', 'device', 'platform', 'setup'
        ]):
            label[7] = 1

        # 9. Product Features
        if any(word in review for word in [
            'feature', 'specification', 'specs', 'capability', 'option', 'setting',
            'configuration', 'customization', 'design', 'functionality', 'built-in',
            'included', 'additional', 'extra', 'advanced', 'basic', 'innovative'
        ]):
            label[8] = 1

        # 10. Others (catch-all category for reviews that don't fit above categories
        # or contain general feedback)
        if (sum(label) == 0) or any(word in review for word in [
            'recommend', 'suggestion', 'feedback', 'general', 'overall',
            'impression', 'thought', 'opinion', 'review', 'comment',
            'miscellaneous', 'other', 'else', 'additional'
        ]):
            label[9] = 1

        labels.append(label)
    return np.array(labels)

def train_model(model, train_loader, val_loader, device, num_epochs=1):
    # Train the BERT model
    optimizer = AdamW(model.parameters(), lr=2e-5)
    criterion = torch.nn.BCEWithLogitsLoss()

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids, attention_mask=attention_mask)
                loss = criterion(outputs.logits, labels)
                val_loss += loss.item()

        print(f'Epoch {epoch + 1}:')
        print(f'Training Loss: {total_loss/len(train_loader):.4f}')
        print(f'Validation Loss: {val_loss/len(val_loader):.4f}')

def evaluate_model(model, test_loader, device):
    # Evaluate the model using various metrics
    model.eval()
    all_preds = []
    all_labels = []
    start_time = time.time()

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            preds = torch.sigmoid(outputs.logits)
            preds = (preds > 0.5).float()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Calculate metrics
    hamming = hamming_loss(all_labels, all_preds)
    subset_acc = accuracy_score(all_labels, all_preds)
    micro_f1 = f1_score(all_labels, all_preds, average='micro')
    macro_f1 = f1_score(all_labels, all_preds, average='macro')

    # Calculate AUC-ROC for each category
    auc_scores = []
    for i in range(all_labels.shape[1]):
        auc = roc_auc_score(all_labels[:, i], all_preds[:, i])
        auc_scores.append(auc)

    inference_time = time.time() - start_time

    return {
        'hamming_loss': hamming,
        'subset_accuracy': subset_acc,
        'micro_f1': micro_f1,
        'macro_f1': macro_f1,
        'auc_scores': auc_scores,
        'inference_time': inference_time
    }

def main():
    # Load and preprocess data
    print("Loading data")
    # json_file_path = os.path.join('data', 'electronics_reviews.json')
    json_file_path = '/content/drive/MyDrive/electronics_reviews.json'
    df = pd.read_json(json_file_path, lines=True)

    # Use only 50000 records for testing
    df = df.sample(n=50000, random_state=42)
    print(f"Using {len(df)} records for testing")

    # Combine review text and summary
    df['combined_text'] = df['reviewText'] + ' ' + df['summary']

    # Clean text
    df['combined_text'] = df['combined_text'].apply(preprocess_text)

    # Create labels
    labels = create_labels(df)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df['combined_text'], labels, test_size=0.2, random_state=42
    )

    # Initialize tokenizer and model
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertForSequenceClassification.from_pretrained(
    'distilbert-base-uncased',
    num_labels=10,
    problem_type="multi_label_classification"
    )

    # Create datasets
    train_dataset = ReviewDataset(X_train, y_train, tokenizer)
    test_dataset = ReviewDataset(X_test, y_test, tokenizer)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8)

    print(f"Training samples: {len(train_dataset)}")
    print(f"Testing samples: {len(test_dataset)}")

    # Train model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    print("Training model...")
    train_model(model, train_loader, test_loader, device)

    # Evaluate model
    print("Evaluating model...")
    metrics = evaluate_model(model, test_loader, device)

    print("\nModel Evaluation Results:")
    print(f"Hamming Loss: {metrics['hamming_loss']:.4f}")
    print(f"Subset Accuracy: {metrics['subset_accuracy']:.4f}")
    print(f"Micro F1 Score: {metrics['micro_f1']:.4f}")
    print(f"Macro F1 Score: {metrics['macro_f1']:.4f}")
    print("\nAUC-ROC Scores for each category:")
    categories = ['Product Quality', 'Customer Service', 'Price', 'Functionality', 'Technical Issues', 'Shipping/Delivery', 'User Experience', 'Product Compatibility', 'Product Features', 'Others']
    for cat, score in zip(categories, metrics['auc_scores']):
        print(f"{cat}: {score:.4f}")
    print(f"\nInference Time: {metrics['inference_time']:.2f} seconds")

    # Save the model
    # print("Saving model")
    # model.save_pretrained('trained_model')
    # tokenizer.save_pretrained('trained_model')
    # print("Model saved successfully!")

    # Save the model to Google Drive
    model_save_path = '/content/drive/MyDrive/trained_model'
    os.makedirs(model_save_path, exist_ok=True)
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    print("Model saved to Google Drive successfully!")

if __name__ == "__main__":
    main()
