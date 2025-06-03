from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

app = Flask(__name__)
CORS(app)  # This allows your API to be called from different origins

# Load your saved model
model_path = "./emotion_bert_model"
if os.path.exists(model_path):
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    
    # Map IDs to emotion names
    id_to_emotion = {0: 'neutral', 1: 'empty', 2: 'love', 3: 'sadness', 
                    4: 'happiness', 5: 'worry', 6: 'relief', 7: 'hate', 
                    8: 'surprise', 9: 'fun', 10: 'boredom', 
                    11: 'enthusiasm', 12: 'anger'}
    print("Model loaded successfully!")
else:
    print(f"Error: Model not found at {model_path}")
    exit(1)

@app.route('/', methods=['GET'])
def home():
    return """
    <h1>Emotion Classification API</h1>
    <p>Send POST requests to /api/predict with JSON data containing 'text' field.</p>
    """

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get('text', '')
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    # Tokenize and predict
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=64)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get probabilities and prediction
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)[0]
    prediction = torch.argmax(outputs.logits, dim=1).item()
    
    # Get top 3 emotions
    top_probs, top_indices = torch.topk(probabilities, 3)
    
    # Create response
    top_emotions = [
        {
            'emotion': id_to_emotion[idx.item()], 
            'confidence': float(prob.item())  # Convert to float for JSON serialization
        } for idx, prob in zip(top_indices, top_probs)
    ]
    
    return jsonify({
        'prediction': id_to_emotion[prediction],
        'confidence': float(probabilities[prediction].item()),
        'top_emotions': top_emotions
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
