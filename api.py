from fastapi import FastAPI
from pydantic import BaseModel
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
import nest_asyncio
import uvicorn
from pyngrok import ngrok

ngrok.set_auth_token("2x5wSYflpa9b37pfbw5gQRHxgYR_WAWzKdUuM6S8UCQ4vrKg")
app = FastAPI()
model_path = '/content/drive/MyDrive/trained_model' 
model = DistilBertForSequenceClassification.from_pretrained(model_path)
tokenizer = DistilBertTokenizer.from_pretrained(model_path)
model.eval()

# Define category labels
categories = [
    'Product Quality', 'Customer Service', 'Price', 'Functionality',
    'Technical Issues', 'Shipping/Delivery', 'User Experience',
    'Product Compatibility', 'Product Features', 'Others'
]

# Request/Response schema
class PredictionRequest(BaseModel):
    text: str
class PredictionResponse(BaseModel):
    categories: dict

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    # Tokenize input text
    inputs = tokenizer(
        request.text,
        add_special_tokens=True,
        max_length=512,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    # Predict
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.sigmoid(outputs.logits)
        predictions = (predictions > 0.5).float()
        print('pred:', predictions, 'out:', outputs);

    # Format result
    result = {
        categories[i]: bool(predictions[0][i])
        for i in range(len(categories))
    }

    return {"categories": result}

# Run the server and expose it with ngrok
nest_asyncio.apply()

# Expose the app
public_url = ngrok.connect(8000)
print(f"FastAPI app is available at: {public_url}/docs")

# Start the app
uvicorn.run(app, host="0.0.0.0", port=8000)
