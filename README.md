# guvi-ml-project

# Multi-Label Classification of Customer Support Tickets with Fine-Tuned BERT

This project uses a DistilBERT model fine-tuned for **multi-label classification** of product reviews into 10 predefined categories. It includes training, evaluation, and deployment using a FastAPI.

## Project Structure

├── main.py - Training, preprocessing, evaluation

├── api.py - FastAPI app for deployment

├── electronics_reviews.json - Dataset (input)

├── trained_model/ - Saved model and tokenizer

## Requirements

Install the necessary packages:

pip install transformers torch fastapi uvicorn nest-asyncio pyngrok pandas scikit-learn

## How to Run

### 1. **Train the Model**

Make sure the dataset is in Google Drive:
```python
python main.py
```

This will:
- Load and preprocess the data
- Train DistilBERT on review classification
- Evaluate using F1, Hamming loss, Subset Accuracy, AUC-ROC
- Save model to `/content/drive/MyDrive/trained_model`

### 2. **Run the API (Colab Compatible)**

```python
python api.py
```

This:
- Starts a FastAPI server
- Exposes a public endpoint using **ngrok**

You’ll get a link like:
```
FastAPI app is available at: https://abc123.ngrok.io/docs
```

Use this UI to test predictions!

---

## 📝 API Usage

### `POST /predict`

**Request:**
```json
{
  "text": "The product quality is good."
}
